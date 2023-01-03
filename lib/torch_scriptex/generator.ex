defmodule TorchScriptex.Generator do
  alias Nx.Tensor, as: T
  import TorchScriptex.Generator.CustomModules

  @prefix "
import torch

class Pad(torch.nn.Module):
    def __init__(self):
        super(Pad, self).__init__()
        self.pad_internal = Pad_Internal()
        self.slice_negative_padding = Slice_Negative_Padding()

    def forward(self, tensor, constant, input_config):
        config = []

        for (a, b, c) in input_config:
            if c < 0:
                raise AttributeError
            config.append([max(a, 0), max(b, 0)])

        config.reverse()
        flat_config = tuple([item for sublist in config for item in sublist])

        tensor = self.pad_internal(tensor, input_config)
        tensor = self.slice_negative_padding(tensor, input_config)
        # input_config = tuple([e for input in input_config for e in input])
        tensor = torch.nn.functional.pad(
            tensor, flat_config, \"constant\", constant)
        return tensor.contiguous()


class Pad_Internal(torch.nn.Module):
    def __init__(self):
        super(Pad_Internal, self).__init__()
        self.torchx_slice = Torchx_Slice()

    def forward(self, t_tx, input_config, pad_value=0):
        pad_sizes = [e[2] for e in input_config]

        if all(size == 0 for size in pad_sizes):
            return t_tx

        pads = []
        for size in reversed(pad_sizes):
            pads.extend([0, size, 0, 0])

        shape = t_tx.shape
        rank = len(shape)
        shape_list = list(shape)
        expanded_shape = tuple([(e, 1) for e in shape_list])

        shape_after_pad = tuple(
            [size + pad * size for size, pad in zip(shape_list, pad_sizes)])

        final_sizes = [size + pad * (size - 1)
                       for size, pad in zip(shape_list, pad_sizes)]

        t_tx = torch.reshape(t_tx, expanded_shape)
        t_tx = torch.pad(t_tx, pads, value=pad_value)
        t_tx = torch.reshape(t_tx, shape_after_pad)
        t_tx = self.torchx_slice(shape_after_pad, tuple(
            final_sizes), rank*[0], final_sizes, rank*[1])

        return t_tx


class Slice_Negative_Padding(torch.nn.Module):
    def __init__(self):
        super(Slice_Negative_Padding, self).__init__()
        self.slice = Torchx_Slice()

    def forward(self, t_tx, input_config):
        if any(pre < 0 or post < 0 for (pre, post, _) in input_config):
            shape = t_tx.shape
            starts, lengths = [], []
            for (axis, (pre, post, _inner)) in enumerate(input_config):
                start = -pre if pre < 0 else 0
                axis_size = shape[axis]
                length = axis_size + post - start if post < 0 else axis_size - start
                starts.append(start)
                lengths.append(length)
            strides = len(shape)*[1]
            return self.slice(t_tx, shape, tuple(lengths), starts, lengths, strides)
        else:
            return t_tx


class Iota(torch.nn.Module):
    def __init__(self):
        super(Iota, self).__init__()

    def forward(self, shape, type=torch.long, axis=0):
        if axis == 0:
            torch.arange(0, shape[0], 1, dtype=type)

        dim = shape[axis]

        aten = torch.arange(0, dim, 1, dtype=type)
        reshape = len(shape)*[1]
        reshape[axis] = dim
        reshape = tuple(reshape)
        aten = torch.reshape(aten, reshape)

        return torch.broadcast_to(aten, shape)


class Slice(torch.nn.Module):
    def __init__(self):
        super(Slice, self).__init__()
        self.shape_slice = Shape_Slice()
        self.torchx_slice = Torchx_Slice()

    def forward(self, tensor, start_indices, lengths, strides=1):
        strides = len(tensor.shape) * \
            [strides] if isinstance(strides, int) else strides
        #print(tensor)
        (start_indices, output_shape) = self.shape_slice(
            tensor.shape, 0, start_indices, lengths, strides, [], [])
        return self.torchx_slice(tensor, tensor.shape, output_shape, start_indices, lengths, strides).contiguous()


class Torchx_Slice(torch.nn.Module):
    def __init__(self):
        super(Torchx_Slice, self).__init__()
        self.narrow = Narrow()
        self.stride = Stride()

    def forward(self, t, input_shape, output_shape, start_indices, lengths, strides):
        t = self.narrow(t, start_indices, lengths, 0, input_shape).contiguous()
        t = self.stride(t, output_shape, lengths, strides)
        return t


class Narrow(torch.nn.Module):
    def __init__(self):
        super(Narrow, self).__init__()

    def forward(self, ref, start_indices, lengths, axis, shape):
        if start_indices == [] and lengths == []:
            return ref
        start = start_indices[0]
        length = lengths[0]

        dim = shape[axis]
        start = min(start, dim - length)
        if start == 0 and length == dim:
            return self.forward(ref, start_indices[1:], lengths[1:], axis+1, shape)
        else:
            ref = torch.narrow(ref, axis, start, length)
            ref = self.forward(
                ref, start_indices[1:], lengths[1:], axis+1, shape)
            return ref


class Stride(torch.nn.Module):
    def __init__(self):
        super(Stride, self).__init__()

    def forward(self, ref, shape, lengths, strides):
        if all(e == 1 for e in strides):
            return ref
        else:
            return torch.as_strided(ref, shape, self.steps_to_strides(lengths, strides), 0)

    def steps_to_strides(self, shape, steps):
        offset = 1
        strides = []
        for (dim, step) in reversed(list(zip(shape, steps))):
            strides = [offset * step] + strides
            offset = offset * dim
        return strides


class Shape_Slice(torch.nn.Module):
    def __init__(self):
        super(Shape_Slice, self).__init__()

    def forward(self, shape, pos, indices, lengths, strides, acc_indices, acc_shape):
        if indices == [] and lengths == [] and strides == []:
            return (list(reversed(acc_indices)), tuple(reversed(acc_shape)))

        i = indices[0]
        len = lengths[0]
        s = strides[0]
        dim = shape[pos]
        out = int(torch.ops.aten.ceil.float(len / s))
        i = min(i, dim - len) if isinstance(i, int) else i
        return self.forward(shape, pos + 1, indices[1:], lengths[1:], strides[1:], [i] + acc_indices, [out] + acc_shape)


class Squeeze(torch.nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, t_tx, axes = None):
        old_shape = t_tx.shape
        axes = self.shape_squeeze_axes(old_shape) if not axes else axes
        axes = self.shape_normalize_axes(old_shape, axes)
        new_shape = self.shape_squeeze(old_shape, axes)

        if old_shape == new_shape:
            return t_tx

        for axis in sorted(axes, reverse=True):
            t_tx = torch.squeeze(t_tx, axis).contiguous()

        return t_tx

    def shape_squeeze_axes(self, shape):
        return [i for (e, i) in list(filter(lambda pair: pair[1] == 1, enumerate(list(shape))))]

    def shape_normalize_axis(self, shape, axis):
        shape_size = len(shape)
        if axis < 0 and abs(axis) <= shape_size:
            return shape_size + axis
        elif axis >= 0 and axis < shape_size:
            return axis

    def shape_normalize_axes(self, shape, axes):
        return [self.shape_normalize_axis(shape, axis) for axis in axes]

    def shape_squeeze(self, shape, axes):
        return self.shape_squeeze_internal(list(enumerate(list(shape))), axes, [])

    def shape_squeeze_internal(self, shape, axes, sacc):
        if len(shape) == 0:
            return tuple(reversed(sacc))

        (s, i) = shape[0]
        if i in axes:
            if s == 1:
                return self.shape_squeeze_internal(shape[1:], axes, sacc)
        else:
            return self.shape_squeeze_internal(shape[1:], axes, [s] + sacc)

class Fun(torch.nn.Module):
    def __init__(self):
        super(Fun, self).__init__()
        self.pad = Pad()
        self.squeeze = Squeeze()
        self.slice = Slice()
        self.iota = Iota()
  "

  @opaque_ops [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
                [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :sign, :abs, :bitwise_not] ++
                [:floor, :ceil, :round] ++
                [:erf, :erfc, :acos, :asin, :atan, :real, :imag] ++
                [:remainder, :atan2, :max, :min] ++
                [:bitwise_and, :bitwise_or, :bitwise_xor] ++
                [:equal, :not_equal, :greater, :less, :less_equal, :greater_equal] ++
                [:logical_and, :logical_or, :logical_xor] ++
                [:add, :subtract, :multiply, :divide]

  @name_exception_ops [:erf_inv, :is_nan, :is_infinity, :left_shift] ++
                        [:negate, :power, :quotient, :right_shift] ++
                        [:concatenate]

  @name_exception_replacements %{
    erf_inv: "erfinv",
    is_nan: "isnan",
    is_infinity: "isinf",
    left_shift: "bitwise_left_shift",
    negate: "neg",
    power: "pow",
    quotient: "div",
    right_shift: "bitwise_right_shift",
    concatenate: "cat"
  }

  @custom_module_ops [:slice, :pad, :squeeze]

  def python(funs, params, consts) do
    IO.inspect(@iota)
    header = prepare_header(params)
    consts = prepare_consts(consts)

    [{_, out, _, _, _} | _] = funs

    body = prepare_body(funs) ++ ["\treturn #{out}"]

    [
      @prefix,
      header,
      consts,
      body
    ]
    |> List.flatten()
    |> Enum.join("\n")
  end

  defp prepare_header(params) do
    stringified_params = params |> Map.keys() |> Enum.sort() |> args_string()
    ["", "    def forward(self, #{stringified_params}):"]
  end

  defp prepare_consts(consts) do
    Enum.map(consts, fn
      {var_name, %Nx.Tensor{} = t} ->
        tensor_repr = t |> Nx.to_flat_list() |> list_to_string()
        shape_repr = t |> Nx.shape() |> python_tuple()
        "#{var_name} = torch.tensor(#{tensor_repr}).reshape(#{shape_repr})"

      {var_name, num} ->
        "#{var_name} = #{num}"
    end)
    |> Enum.map(&indent/1)
  end

  defp prepare_body(funs) do
    funs
    |> Enum.reverse()
    |> to_lines()
    |> Enum.reverse()
  end

  defp to_lines(lines \\ [], line)

  defp to_lines(lines, [{tabs, var_name, op, args}]) do
    torch_assignment(var_name, op, args)
    |> indent(tabs + 1)
    |> then(&[&1 | lines])
  end

  defp to_lines(lines, [{tabs, out, :while, [vars, names]}, condition | tail]) do
    {_, _, condition_op, condition_args, _} = condition

    variables =
      [vars, names]
      |> Enum.zip()
      |> Enum.map(fn {var, name} -> indent("#{name} = #{var}", tabs + 1) end)

    return = [indent("#{out} = [#{Enum.join(Enum.reverse(names), (","))}]", tabs + 1)]

    while_header(condition_op, condition_args)
    |> indent(tabs + 1)
    |> List.wrap()
    |> Enum.concat(return ++ variables ++ lines)
    |> to_lines(tail)
  end

  defp to_lines(lines, [{tabs, var_name, op, args} | tail]) do
    torch_assignment(var_name, op, args)
    |> indent(tabs + 1)
    |> then(&[&1 | lines])
    |> to_lines(tail)
  end

  defp to_lines(lines, [{tabs, var_name, op, args, out}]) do
    torch_assignment(var_name, op, args, out)
    |> indent(tabs + 1)
    |> then(&[&1 | lines])
  end

  defp to_lines(lines, [{tabs, var_name, op, args, out} | tail]) do
    torch_assignment(var_name, op, args, out)
    |> indent(tabs + 1)
    |> then(&[&1 | lines])
    |> to_lines(tail)
  end

  defp torch_assignment(var_name, op, args) do
    "#{var_name} = #{torch_op(op, args)}"
  end

  defp torch_assignment(var_name, op, args, out) do
    "#{var_name} = #{torch_op(op, args, out)}"
  end

  defp while_header(condition_op, condition_args) do
    "while(#{torch_op(condition_op, wrap_tensor(condition_args))}):"
  end

  defp torch_op(op, args) when op in @opaque_ops do
    "torch.#{op}(#{args_string(args)})"
  end

  defp torch_op(op, args) when op in @name_exception_ops do
    "torch.#{@name_exception_replacements[op]}(#{args_string(args)})"
  end

  defp torch_op(op, args, _) when op in @opaque_ops do
    "torch.#{op}(#{args_string(args)})"
  end

  defp torch_op(op, args, _) when op in @name_exception_ops do
    "torch.#{@name_exception_replacements[op]}(#{args_string(args)})"
  end

  defp torch_op(op, args) when op in @custom_module_ops do
    "self.#{op}(#{args_string(args)})"
  end

  defp torch_op(op, args, _) when op in @custom_module_ops do
    "self.#{op}(#{args_string(args)})"
  end

  defp torch_op(:assign, args) when length(args) > 1, do: "#{list_to_string(args)}"

  defp torch_op(:assign, [arg]), do: "#{arg}"

  defp torch_op(:multiple_assign, args), do: "#{Enum.join(args, ",")}"

  defp torch_op(:elem, [arg, index]), do: "#{arg}[#{index}]"

  defp torch_op(:iota, _, %T{shape: shape}), do: "self.iota(#{python_tuple(shape)})"

  defp torch_op(:cbrt, [arg]), do: "torch.pow(#{arg}, 1.0/3)"

  defp torch_op(:conjugate, [arg]), do: "torch.conj(#{arg}).resolve_conj()"

  defp torch_op(:list, args), do: "[#{Enum.reverse(args) |> Enum.join(",")}]"

  defp torch_op(:reshape, [arg], %T{shape: shape}), do: "torch.reshape(#{arg}, #{python_tuple(shape)})"

  defp torch_op(:as_type, [arg], %T{type: type}), do: "#{arg}.type(#{python_type(type)})"

  #undefined behaviour in torchscript, bitcast() unsupported
  defp torch_op(:bitcast, [arg], %T{type: type}), do: "#{arg}.view(#{python_type(type)})"

  defp torch_op(op, args) do
    "#{op}(#{Enum.join(args,",")}) not supported"
    #raise(ArgumentError, "#{op} Nx operation is unsupported for generating TorchScript code")
  end

  defp torch_op(op, args, out) do
    "#{op}(#{Enum.join(args,",")}) not supported"
    #raise(ArgumentError, "#{op} Nx operation is unsupported for generating TorchScript code")
  end

  # String helpers

  defp wrap_tensor(arg) when is_list(arg), do: Enum.map(arg, fn arg -> "torch.tensor(" <> arg <> ")" end)
  defp wrap_tensor(arg), do: "torch.tensor(" <> arg <> ")"

  defp args_string(args), do: Enum.reverse(args) |> Enum.join(", ")

  defp indent(string, tabs \\ 1) do
    "#{String.duplicate("\t", tabs)}#{string}"
  end

  defp list_to_string(list) when is_list(list) do
    "[#{Enum.map_join(list, ", ", &list_to_string/1)}]"
  end

  defp list_to_string(elem), do: elem

  defp python_tuple({elem}), do: "(#{elem},)"

  defp python_tuple(tuple) when is_tuple(tuple) do
    tuple_string = tuple |> Tuple.to_list() |> Enum.join(", ")
    case tuple_string do
      "" -> "(1,)"
      _ -> "(#{tuple_string})"
    end
  end

  defp python_type(type) do
    case Nx.Type.normalize!(type) do
      {:u, 8} -> "torch.uint8"
      {:s, 8} -> "torch.int8"
      {:s, 16} -> "torch.short"
      {:s, 32} -> "torch.int"
      {:s, 64} -> "torch.long"
      {:f, 32} -> "torch.float"
      {:f, 64} -> "torch.double"
      {:u, _} -> "torch.int"
    end
  end
end
