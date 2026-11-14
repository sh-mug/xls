// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xls/jit/jit_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/DataLayout.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/Support/Alignment.h"
#include "xls/common/bits_util.h"
#include "xls/common/math_util.h"
#include "xls/ir/bits.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/jit/type_buffer_metadata.h"

namespace xls {
namespace {

// Writes `bits` in little-endian byte order to `dst`
// (`CeilOfRatio(bit_count, kCharBit)` bytes) and zeroes any unused padding bits
// in the most-significant byte of value data.
//
// `Bits` types are stored in the JIT as the next power-of-two integer size, and
// LLVM requires all padding bits above `bit_count` to be zero for safe
// operation (e.g., for a 42-bit type padded out to a 64-bit word in the JIT,
// bits 42 through 63 must be zero). Callers clear any trailing container bytes
// beyond the returned `data_byte_count` (either via a tail `memset` in the
// scalar `BlitValueToBuffer` fast path or by zeroing the enclosing composite
// buffer prior to calling `BlitValueImpl`), so only the remainder bits in the
// most-significant byte of value data need to be masked here.
int64_t WriteLittleEndianBitsWithZeroedRemainder(const Bits& bits,
                                                 uint8_t* dst) {
  int64_t bit_count = bits.bit_count();
  int64_t data_byte_count = CeilOfRatio(bit_count, kCharBit);
  if (data_byte_count > 0) {
    bits.ToBytes(absl::MakeSpan(dst, data_byte_count));
    int remainder_bits = bit_count % kCharBit;
    if (remainder_bits != 0) {
      dst[data_byte_count - 1] &= static_cast<uint8_t>(Mask(remainder_bits));
    }
  }
  return data_byte_count;
}

}  // namespace

/* static */ void JitRuntime::MaybeInitCompositeChildMemo(
    const Type* child_type, TypeMetricsMemo* memo) {
  if (!child_type->IsBits() && !child_type->IsToken() && !memo->has_value()) {
    memo->emplace();
  }
}

/* static */ JitRuntime::TargetLayoutInfo
JitRuntime::GetOrComputeTargetLayoutInfo(const llvm::DataLayout& data_layout) {
  // Probing `llvm::DataLayout` requires constructing and querying a temporary
  // `LLVMContext`, which is roughly two orders of magnitude slower than a
  // hash-map lookup and would add noticeable overhead to lightweight AOT
  // runtime instantiation (`CreateFromAot`). Cache the probed
  // `TargetLayoutInfo` by `getStringRepresentation()`; `llvm::DataLayout` is
  // not directly hashable, and equivalent non-canonical layout strings at worst
  // populate duplicate identical cache entries.
  ABSL_CONST_INIT static absl::Mutex cache_mutex(absl::kConstInit);
  static absl::NoDestructor<
      absl::flat_hash_map<std::string, std::unique_ptr<const TargetLayoutInfo>>>
      cache;

  const std::string& layout_str = data_layout.getStringRepresentation();
  {
    absl::MutexLock lock(cache_mutex);
    if (auto it = cache->find(layout_str); it != cache->end()) {
      return *it->second;
    }
  }

  // Probe a local copy of `data_layout` with a temporary `LLVMContext` so that
  // `this->data_layout_` never populates its internal mutable `LayoutMap`.
  llvm::LLVMContext probe_ctx;
  llvm::DataLayout probe_dl = data_layout;
  TargetLayoutInfo info;
  info.is_little_endian = probe_dl.isLittleEndian();

  llvm::StructType* empty_struct = llvm::StructType::get(probe_ctx, {});
  info.struct_abi_alignment = probe_dl.getABITypeAlign(empty_struct).value();
  info.struct_pref_alignment = probe_dl.getPrefTypeAlign(empty_struct).value();

  llvm::Type* token_type =
      llvm::ArrayType::get(llvm::IntegerType::get(probe_ctx, 1), 0);
  info.token_abi_alignment = probe_dl.getABITypeAlign(token_type).value();
  info.token_pref_alignment = probe_dl.getPrefTypeAlign(token_type).value();

  for (int log2_bits = 3; log2_bits <= kMaxLog2Bits; ++log2_bits) {
    llvm::IntegerType* int_ty =
        llvm::IntegerType::get(probe_ctx, uint32_t{1} << log2_bits);
    info.int_abi_alignments[log2_bits] =
        probe_dl.getABITypeAlign(int_ty).value();
    info.int_pref_alignments[log2_bits] =
        probe_dl.getPrefTypeAlign(int_ty).value();
  }

  absl::MutexLock lock(cache_mutex);
  auto [it, inserted] = cache->try_emplace(
      layout_str, std::make_unique<const TargetLayoutInfo>(info));
  return *it->second;
}

JitRuntime::JitRuntime(llvm::DataLayout data_layout)
    : data_layout_(std::move(data_layout)),
      layout_info_(GetOrComputeTargetLayoutInfo(data_layout_)) {}

JitRuntime::TypeMetrics JitRuntime::GetBitsTypeMetrics(
    int64_t bit_count) const {
  int log2_bits = (bit_count <= 8) ? 3 : CeilOfLog2(bit_count);
  CHECK_LE(log2_bits, kMaxLog2Bits)
      << "Unsupported XLS bit count for LLVM IntegerType: " << bit_count;
  int64_t llvm_bytes = int64_t{1} << (log2_bits - 3);
  int64_t abi_align = layout_info_.int_abi_alignments[log2_bits];
  int64_t pref_align = layout_info_.int_pref_alignments[log2_bits];
  int64_t alloc_size = static_cast<int64_t>(
      llvm::alignTo(static_cast<uint64_t>(llvm_bytes), llvm::Align(abi_align)));
  return TypeMetrics{
      .size = alloc_size, .abi_align = abi_align, .pref_align = pref_align};
}

JitRuntime::TypeMetrics JitRuntime::ComputeTypeMetrics(
    const Type* xls_type) const {
  TypeMetricsMemo memo;
  return ComputeTypeMetricsImpl(xls_type, &memo);
}

JitRuntime::TypeMetrics JitRuntime::ComputeTypeMetricsImpl(
    const Type* xls_type, TypeMetricsMemo* memo) const {
  switch (xls_type->kind()) {
    case TypeKind::kBits:
      return GetBitsTypeMetrics(xls_type->AsBitsOrDie()->bit_count());
    case TypeKind::kToken:
      return TypeMetrics{.size = 0,
                         .abi_align = layout_info_.token_abi_alignment,
                         .pref_align = layout_info_.token_pref_alignment};
    case TypeKind::kArray: {
      if (memo->has_value()) {
        if (auto it = (*memo)->find(xls_type); it != (*memo)->end()) {
          return it->second;
        }
      }
      const ArrayType* array_type = xls_type->AsArrayOrDie();
      const Type* elem_type = array_type->element_type();
      MaybeInitCompositeChildMemo(elem_type, memo);
      TypeMetrics elem = ComputeTypeMetricsImpl(elem_type, memo);
      TypeMetrics result{.size = array_type->size() * elem.size,
                         .abi_align = elem.abi_align,
                         .pref_align = elem.pref_align};
      if (memo->has_value()) {
        (*memo)->emplace(xls_type, result);
      }
      return result;
    }
    case TypeKind::kTuple: {
      if (memo->has_value()) {
        if (auto it = (*memo)->find(xls_type); it != (*memo)->end()) {
          return it->second;
        }
      }
      const TupleType* tuple_type = xls_type->AsTupleOrDie();
      uint64_t struct_size = 0;
      int64_t max_elem_abi_align = 1;
      for (const Type* elem_type : tuple_type->element_types()) {
        MaybeInitCompositeChildMemo(elem_type, memo);
        TypeMetrics elem = ComputeTypeMetricsImpl(elem_type, memo);
        struct_size = llvm::alignTo(struct_size, llvm::Align(elem.abi_align));
        max_elem_abi_align = std::max(max_elem_abi_align, elem.abi_align);
        struct_size += static_cast<uint64_t>(elem.size);
      }
      int64_t abi_align =
          std::max(layout_info_.struct_abi_alignment, max_elem_abi_align);
      int64_t pref_align =
          std::max(layout_info_.struct_pref_alignment, max_elem_abi_align);
      struct_size = llvm::alignTo(struct_size, llvm::Align(abi_align));
      TypeMetrics result{.size = static_cast<int64_t>(struct_size),
                         .abi_align = abi_align,
                         .pref_align = pref_align};
      if (memo->has_value()) {
        (*memo)->emplace(xls_type, result);
      }
      return result;
    }
  }
  LOG(FATAL) << "Unsupported XLS Type kind: " << xls_type->kind();
}

int64_t JitRuntime::GetTypeByteSize(const Type* xls_type) const {
  return ComputeTypeMetrics(xls_type).size;
}

int64_t JitRuntime::GetTypeAlignment(const Type* xls_type) const {
  return ComputeTypeMetrics(xls_type).pref_align;
}

int64_t JitRuntime::GetTypeAbiAlignment(const Type* xls_type) const {
  return ComputeTypeMetrics(xls_type).abi_align;
}

TypeBufferMetadata JitRuntime::GetTypeBufferMetadata(
    const Type* xls_type) const {
  TypeMetrics metrics = ComputeTypeMetrics(xls_type);
  int64_t packed_llvm_width = std::max(
      int64_t{1}, RoundUpToNearest(xls_type->GetFlatBitCount(), int64_t{8}));
  return TypeBufferMetadata{
      .size = metrics.size,
      .preferred_alignment = metrics.pref_align,
      .abi_alignment = metrics.abi_align,
      .packed_size = RoundUpToNearest(packed_llvm_width, int64_t{8}),
  };
}

absl::Status JitRuntime::PackArgs(
    absl::Span<const Value> args, absl::Span<Type* const> arg_types,
    absl::Span<uint8_t* const> arg_buffers) const {
  if (arg_buffers.size() < args.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input buffer is not large enough to hold all arguments: %d vs. %d",
        arg_buffers.size(), args.size()));
  }
  for (size_t i = 0; i < args.size(); ++i) {
    const Type* type = arg_types[i];
    if (type->IsBits()) {
      int64_t bit_count = type->AsBitsOrDie()->bit_count();
      int64_t type_size = GetBitsTypeMetrics(bit_count).size;
      BlitValueToBuffer(args[i], type,
                        absl::MakeSpan(arg_buffers[i], type_size));
    } else {
      // Pass `memo` into `BlitValueImpl` so nested composite subtype metrics
      // cached during sizing are reused (`BlitValueImpl` computes any missing
      // entry on miss).
      TypeMetricsMemo memo;
      TypeMetrics metrics = ComputeTypeMetricsImpl(type, &memo);
      if (metrics.size > 0) {
        std::memset(arg_buffers[i], 0, metrics.size);
        BlitValueImpl(args[i], type, arg_buffers[i], &memo);
      }
    }
  }
  return absl::OkStatus();
}

Value JitRuntime::UnpackBuffer(const uint8_t* buffer,
                               const Type* result_type) const {
  TypeMetricsMemo memo;
  return UnpackBufferImpl(buffer, result_type, &memo);
}

Value JitRuntime::UnpackBufferImpl(const uint8_t* buffer,
                                   const Type* result_type,
                                   TypeMetricsMemo* memo) const {
  switch (result_type->kind()) {
    case TypeKind::kBits: {
      const BitsType* bits_type = result_type->AsBitsOrDie();
      int64_t bit_count = bits_type->bit_count();
      // `Bits::FromBytes` and `InlineBitmap::FromBytes` are inline in headers:
      // specializing common scalar widths passes a compile-time constant
      // `bit_count` and span size so the compiler folds `InlinedVector` to its
      // single inline word, eliminates `CHECK_GE`, lowers `memcpy` to a single
      // scalar load instead of a library call, and folds `MaskLastWord()` to a
      // constant mask (several times faster than variable `byte_count`).
      switch (bit_count) {
        case 0:
          return Value(Bits(0));
        case 1:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 1), 1));
        case 8:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 1), 8));
        case 16:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 2), 16));
        case 32:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 4), 32));
        case 64:
          return Value(Bits::FromBytes(absl::MakeSpan(buffer, 8), 64));
        default: {
          int64_t byte_count = CeilOfRatio(bit_count, kCharBit);
          return Value(
              Bits::FromBytes(absl::MakeSpan(buffer, byte_count), bit_count));
        }
      }
    }
    case TypeKind::kTuple: {
      const TupleType* tuple_type = result_type->AsTupleOrDie();
      std::vector<Value> values;
      values.reserve(tuple_type->size());
      uint64_t offset = 0;
      for (int64_t i = 0; i < tuple_type->size(); ++i) {
        const Type* elem_type = tuple_type->element_type(i);
        MaybeInitCompositeChildMemo(elem_type, memo);
        TypeMetrics elem_metrics = ComputeTypeMetricsImpl(elem_type, memo);
        offset = llvm::alignTo(offset, llvm::Align(elem_metrics.abi_align));
        values.push_back(UnpackBufferImpl(buffer + offset, elem_type, memo));
        offset += static_cast<uint64_t>(elem_metrics.size);
      }
      return Value::TupleOwned(std::move(values));
    }
    case TypeKind::kArray: {
      const ArrayType* array_type = result_type->AsArrayOrDie();
      if (array_type->empty()) {
        return Value::ArrayOwned({});
      }
      const Type* element_type = array_type->element_type();
      MaybeInitCompositeChildMemo(element_type, memo);
      int64_t elem_size = ComputeTypeMetricsImpl(element_type, memo).size;
      std::vector<Value> values;
      values.reserve(array_type->size());
      for (int64_t i = 0; i < array_type->size(); ++i) {
        values.push_back(
            UnpackBufferImpl(buffer + i * elem_size, element_type, memo));
      }
      return Value::ArrayOwned(std::move(values));
    }
    case TypeKind::kToken:
      return Value::Token();
  }
  LOG(FATAL) << "Unsupported XLS Value kind: " << result_type->kind();
}

void JitRuntime::BlitValueToBuffer(const Value& value, const Type* type,
                                   absl::Span<uint8_t> buffer) const {
  if (type->IsBits()) {
    // Underlying `Bits` object relies on little-endianness.
    CHECK(layout_info_.is_little_endian);
    const Bits& bits = value.bits();
    int64_t type_byte_size = GetBitsTypeMetrics(bits.bit_count()).size;
    int64_t data_byte_count =
        WriteLittleEndianBitsWithZeroedRemainder(bits, buffer.data());
    // Clear any remaining upper bytes of the power-of-two LLVM integer
    // container (the scalar fast path skips the full-buffer `memset` below).
    if (type_byte_size > data_byte_count) {
      std::memset(buffer.data() + data_byte_count, 0,
                  type_byte_size - data_byte_count);
    }
    return;
  }

  // Pass `memo` into `BlitValueImpl` so nested composite subtype metrics
  // populated while computing `type_byte_size` are reused rather than
  // recomputed (`BlitValueImpl` does not require `memo` to be pre-populated for
  // correctness and computes any uncached entry on miss).
  TypeMetricsMemo memo;
  int64_t type_byte_size = ComputeTypeMetricsImpl(type, &memo).size;
  if (type_byte_size > 0) {
    // Zero the buffer before filling in values. This ensures all struct
    // alignment holes and padding bytes are cleared.
    std::memset(buffer.data(), 0, type_byte_size);
    BlitValueImpl(value, type, buffer.data(), &memo);
  }
}

absl::Span<uint8_t> JitRuntime::AsAligned(absl::Span<uint8_t> buffer,
                                          int64_t alignment) const {
  return buffer.subspan(llvm::offsetToAlignment(
      reinterpret_cast<uintptr_t>(buffer.data()), llvm::Align(alignment)));
}

void JitRuntime::BlitValueImpl(const Value& value, const Type* type,
                               uint8_t* buffer, TypeMetricsMemo* memo) const {
  if (value.IsBits()) {
    // Underlying `Bits` object relies on little-endianness. The enclosing
    // composite buffer is zero-initialized before `BlitValueImpl`, so only the
    // remainder bits in the most-significant byte of value data need masking.
    CHECK(layout_info_.is_little_endian);
    WriteLittleEndianBitsWithZeroedRemainder(value.bits(), buffer);
    return;
  }
  if (value.IsArray()) {
    if (value.empty()) {
      return;
    }
    const Type* elem_type = type->AsArrayOrDie()->element_type();
    MaybeInitCompositeChildMemo(elem_type, memo);
    int64_t elem_size = ComputeTypeMetricsImpl(elem_type, memo).size;
    for (int64_t i = 0; i < value.size(); ++i) {
      BlitValueImpl(value.element(i), elem_type, buffer + i * elem_size, memo);
    }
    return;
  }
  if (value.IsTuple()) {
    const TupleType* tuple_type = type->AsTupleOrDie();
    uint64_t offset = 0;
    for (int64_t i = 0; i < value.size(); ++i) {
      const Type* elem_type = tuple_type->element_type(i);
      MaybeInitCompositeChildMemo(elem_type, memo);
      TypeMetrics elem_metrics = ComputeTypeMetricsImpl(elem_type, memo);
      offset = llvm::alignTo(offset, llvm::Align(elem_metrics.abi_align));
      if (elem_metrics.size > 0) {
        BlitValueImpl(value.element(i), elem_type, buffer + offset, memo);
      }
      offset += static_cast<uint64_t>(elem_metrics.size);
    }
    return;
  }
  if (value.IsToken()) {
    return;
  }
  LOG(FATAL) << "Unsupported XLS Value kind: " << value.kind();
}

}  // namespace xls
