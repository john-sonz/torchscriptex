defmodule TorchScriptex.Generator do
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
                        [:negate, :power, :quotient, :right_shift]

  @name_exception_replacements %{
    erf_inv: "erfinv",
    is_nan: "isnan",
    is_infinity: "isinf",
    left_shift: "bitwise_left_shift",
    negate: "neg",
    power: "pow",
    quotient: "div",
    right_shift: "bitwise_right_shift"
  }

  def python(funs, params, consts) do
    header = prepare_header(params)
    consts = prepare_consts(consts)
    body = prepare_body(funs)

    [
      header,
      consts,
      body
    ]
    |> List.flatten()
    |> Enum.join("\n")
  end

  defp prepare_header(params) do
    stringified_params = params |> Map.keys() |> Enum.sort() |> args_string()
    ["@torch.jit.script", "def forward(#{stringified_params}):"]
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

  defp to_lines(lines, [{tabs, _, :while, [vars, names]}, condition | tail]) do
    {_, _, condition_op, condition_args} = condition

    variables =
      [vars, names]
      |> Enum.zip()
      |> Enum.map(fn {var, name} -> indent("#{name} = #{var}", tabs + 1) end)

    while_header(condition_op, condition_args)
    |> indent(tabs + 1)
    |> List.wrap()
    |> Enum.concat(variables ++ lines)
    |> to_lines(tail)
  end

  defp to_lines(lines, [{tabs, var_name, op, args} | tail]) do
    torch_assignment(var_name, op, args)
    |> indent(tabs + 1)
    |> then(&[&1 | lines])
    |> to_lines(tail)
  end

  defp torch_assignment(var_name, op, args) do
    "#{var_name} = #{torch_op(op, args)}"
  end

  defp while_header(condition_op, condition_args) do
    "while(#{torch_op(condition_op, condition_args)}):"
  end

  defp torch_op(op, args) when op in @opaque_ops do
    "torch.#{op}(#{args_string(args)})"
  end

  defp torch_op(op, args) when op in @name_exception_ops do
    "torch.#{@name_exception_replacements[op]}(#{args_string(args)})"
  end

  defp torch_op(:elem, [arg, index]), do: "#{arg}[#{index}]"

  defp torch_op(:cbrt, [arg]), do: "torch.pow(#{arg}, 1.0/3)"

  defp torch_op(:conjugate, [arg]), do: "torch.conj(#{arg}).resolve_conj()"

  defp torch_op(op, _) do
    raise(ArgumentError, "#{op} Nx operation is unsupported for generating TorchScript code")
  end

  # String helpers

  defp args_string(args), do: Enum.join(args, ", ")

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
    "(#{tuple_string})"
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
      {:u, _} -> "torch.long"
    end
  end
end
