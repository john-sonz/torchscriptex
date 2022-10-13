defmodule TorchScriptex do
  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  @enforce_keys [:id, :op, :args, :context]
  defstruct [:id, :op, :args, :context]

  import Nx.Defn

  defn softmax(t, p, r) do
    ret = Nx.add(t, p)
    |> Nx.subtract(r)
    |> Nx.multiply(2)
    |> Nx.divide(4)
    inspectx(ret)
  end

  deftransform inspectx(tensor) do
    {t, acc} = inspect_expr(tensor, {[], [], %{}, %{}})
    {_, {exprs, params, _var_map, _cache}} = Tree.apply_args(t, acc, &inspect_expr/2)

    params = Enum.reverse(params) |> Enum.uniq()
    exprs = Enum.reverse(exprs) |> Enum.uniq()
    header = Enum.reduce(params, "def forward(self", fn {_, p}, acc ->
      acc <> ", " <> p
    end) <> "):"

    {expr} = List.last(exprs)
    [returned | _] = String.split(expr, " ")

    all = Enum.reduce(exprs, header, fn {expr}, acc ->
      acc <> "\n\t" <> expr
    end) <> "\n\treturn " <> returned


    IO.puts(all)
    tensor
  end

  # Constants and funs are shown as is
  defp inspect_expr(%T{data: %Expr{op: :constant}} = t, acc), do: {t, acc}
  defp inspect_expr(%T{data: %Expr{op: :fun}} = t, acc), do: {t, acc}

  defp inspect_expr(%T{data: %Expr{op: :metadata, args: [expr, metadata]}}, acc)
       when not is_map_key(metadata, :inspect),
       do: inspect_expr(expr, acc)

  defp inspect_expr(%T{data: %Expr{op: :optional, args: [expr, _default]}}, acc) do
    inspect_expr(expr, acc)
  end

  defp inspect_expr(%T{data: %Expr{id: id}} = t, {exprs, params, var_map, cache} = acc) do
    case cache do
      %{^id => _} -> {t, acc}
      %{} -> cached_inspect_expr(t, {exprs, params, var_map, Map.put(cache, id, true)})
    end
  end

  defp cached_inspect_expr(%T{data: %Expr{op: :parameter, id: id, args: _}} = t, acc) do
    {exprs, params, var_map, cache} = acc
    {var, var_map} = var_for_id(var_map, id)
    param = {"parameter", var}
    {t, {exprs, [param | params], var_map, cache}}
  end

  defp cached_inspect_expr(%T{data: %Expr{op: :tensor, id: id}} = t, acc) do
    {exprs, params, var_map, cache} = acc
    {var, var_map} = var_for_id(var_map, id)
    param = {"tensor ", var}
    {t, {exprs, [{param} | params], var_map, cache}}
  end

  defp cached_inspect_expr(%T{data: %Expr{id: id, op: op}} = t, acc) do
    {args, {exprs, params, var_map, cache}} = traverse_args(op, t, acc)
    {var, var_map} = var_for_id(var_map, id)
    args_str = inspect_args(op, args, var_map)
    expr_str = var <> " = " <> pythonize(op, args_str)
    {t, {[{expr_str} | exprs], params, var_map, cache}}
  end

  defp pythonize(:add, args_str) do
    binary_glue("torch.add", args_str)
  end

  defp pythonize(:subtract, args_str) do
    binary_glue("torch.subtract", args_str)
  end

  defp pythonize(:multiply, args_str) do
    binary_glue("torch.multiply", args_str)
  end

  defp pythonize(:divide, args_str) do
    binary_glue("torch.divide", args_str)
  end

  defp pythonize(op, args_str) do
    Atom.to_string(op) <> " " <> args_str
  end

  defp binary_glue(op, args_str) do
    op <> "(" <> args_str <> ")"
  end




























  defp traverse_args(:while, %T{data: %{args: [initial, _, _, _]}}, acc) do
    {initial, acc} = Composite.traverse(initial, acc, &inspect_expr/2)
    {[initial], acc}
  end

  defp traverse_args(:token, %T{data: %{args: [token]}}, acc) do
    {hooks, acc} =
      Enum.map_reduce(token.hooks, acc, fn %{name: name, expr: expr}, acc ->
        {expr, acc} = Composite.traverse(expr, acc, &inspect_expr/2)
        {{name, expr}, acc}
      end)

    {hooks, acc}
  end

  defp traverse_args(_op, t, acc) do
    Tree.apply_args(t, acc, &inspect_expr/2)
  end

  defp inspect_args(:token, hooks, var_map) do
    IO.iodata_to_binary(
      Enum.map_intersperse(hooks, ", ", fn {key, val} ->
        "#{key}: " <> inspect_arg(val, var_map)
      end)
    )
  end

  defp inspect_args(:while, [initial], var_map) do
    IO.iodata_to_binary(inspect_arg(initial, var_map))
  end

  defp inspect_args(:cond, [clauses, last], var_map) do
    clauses =
      Enum.map(clauses, fn {pred, expr} ->
        [inspect_arg(pred, var_map), " -> ", inspect_arg(expr, var_map), ", "]
      end)

    IO.iodata_to_binary([clauses, "true -> ", inspect_arg(last, var_map)])
  end

  defp inspect_args(:metadata, [expr, %{inspect: inspect}], var_map) do
    IO.iodata_to_binary([inspect_arg(expr, var_map), ", ", inspect(inspect)])
  end

  defp inspect_args(_op, [tuple | args], var_map) when is_tuple(tuple),
    do: inspect_args(args, var_map)

  defp inspect_args(_op, args, var_map),
    do: inspect_args(args, var_map)

  defp inspect_args(args, var_map),
    do: Enum.map_join(args, ", ", &inspect_arg(&1, var_map))

  defp inspect_arg(arg, var_map) do
    case arg do
      %T{data: %Expr{op: :fun, args: [_, _, {m, f, a}]}} ->
        [?&, Exception.format_mfa(m, f, a)]

      %T{data: %Expr{op: :constant, args: [number]}} ->
        to_string(number)

      %T{data: %Expr{id: id}} ->
        Map.fetch!(var_map, id)

      _ ->
        cond do
          Keyword.keyword?(arg) and arg != [] ->
            Enum.map_join(arg, ", ", fn {k, v} -> "#{k}: #{inspect(v)}" end)

          is_list(arg) ->
            [?[, inspect_args(arg, var_map), ?]]

          is_tuple(arg) ->
            [?{, inspect_args(Tuple.to_list(arg), var_map), ?}]

          true ->
            inspect(arg)
        end
    end
  end

  defp var_for_id(var_map, id) do
    case var_map do
      %{^id => var} ->
        {var, var_map}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, Map.put(var_map, id, var)}
    end
  end

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [Enum.at(?a..?z, counter)]
end
