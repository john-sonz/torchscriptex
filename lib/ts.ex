defmodule TS do
  alias Nx.Defn.{Expr, Tree}
  alias Nx.Tensor, as: T

  @enforce_keys [:id, :op, :args]
  defstruct [:id, :op, :args]

  import Nx.Defn

  def inspect_torch(var_or_vars, fun) when is_function(fun, 1) do
    jit_apply(
      fn var_or_vars ->
        var_or_vars
        |> fun.()
        |> inspectx()
      end,
      [var_or_vars],
      on_conflict: :reuse
    )
  end

  defn softmax(x) do
    {_, x} =
      while {i = 0, x}, Nx.less(i, 5) do
        {i + 1, x + i}
      end

    Nx.exp(x) / Nx.sum(Nx.exp(x))
  end

  defn slice_test() do
    t =
      Nx.tensor([
        [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15, 16, 17]],
        [[71, 72, 73, 74, 75, 76, 77, 78], [81, 82, 83, 84, 85, 86, 87, 88]]
      ])

    Nx.slice(t, [2, 0, 1], [2, 1, 2], strides: [1, 1, 2])
    |> print_expr()
    |> inspectx()

    # TorchScriptex.inspect_torch({t}, fn {a} -> Nx.slice(a,[2,0,1],[2,1,2],strides: [1,2,3]) end)
    # Nx.Defn.
  end

  def tensor_test(a, b, c) do
    # Nx.LinAlg.cholesky(Nx.tensor([[20.0, 17.6], [17.6, 16.0]]))
    TS.inspect_torch(Nx.tensor([[20.0, 17.6], [17.6, 16.0]]), fn x -> Nx.LinAlg.cholesky(x) end)
    # TS.inspect_torch({a, b, c}, fn {a, b, c} -> Nx.Random.uniform_split(a, b, c) end)
  end

  deftransform inspectx(tensor) do
    {t, acc} = inspect_expr(tensor, {[], [], %{}, %{}})
    {_, {expr, params, _var_map, _cache}} = Tree.apply_args(t, acc, &inspect_expr/2)

    params = Enum.reverse(params) |> Enum.uniq()
    IO.inspect(params, label: "PARAMS")
    IO.inspect(t, label: "TENSOR")
    IO.inspect(expr, label: "EXPR")

    expr
    |> to_python()
    |> Enum.reverse()
    |> Enum.join("\n")
    |> IO.puts()

    tensor
  end

  def to_python(expr, acc \\ [])

  def to_python(%TS{op: op}, acc) when op in [:tensor, :parameter] do
    acc
  end

  def to_python(%TS{id: id, op: :concatenate = op, args: [list | args]}, acc)
      when is_list(list) do
    acc = Enum.reduce(list ++ args, acc, &to_python/2)

    list_str =
      Enum.map_join(list, ",", fn
        %TS{id: id} -> id
        arg -> inspect(arg)
      end)

    args_str =
      Enum.map_join(args, ",", fn
        %TS{id: id} -> id
        arg -> inspect(arg)
      end)

    ["#{id} = #{op}([#{list_str}], #{args_str})" | acc]
  end

  def to_python(%TS{id: id, op: op, args: args}, acc) do
    acc = Enum.reduce(args, acc, &to_python/2)

    args_str =
      Enum.map_join(args, ",", fn
        %TS{id: id} -> id
        arg -> inspect(arg)
      end)

    ["#{id} = #{op}(#{args_str})" | acc]
  end

  def to_python(_, acc), do: acc

  # Constants and funs are shown as is
  defp inspect_expr(%T{data: %Expr{op: :constant}} = t, acc), do: {t, acc}
  defp inspect_expr(%T{data: %Expr{op: :fun}} = t, acc), do: {t, acc}

  defp inspect_expr(%T{data: %Expr{op: :metadata, args: [expr, metadata]}}, acc)
       when not is_map_key(metadata, :inspect),
       do: inspect_expr(expr, acc)

  defp inspect_expr(%T{data: %Expr{op: :optional, args: [expr, _default]}}, acc) do
    inspect_expr(expr, acc)
  end

  defp inspect_expr(%T{data: %Expr{id: id}} = t, {_, _, _, cache} = acc) do
    case cache do
      %{^id => _} -> {t, acc}
      %{} -> cached_inspect_expr(t, acc)
    end
  end

  defp cached_inspect_expr(%T{data: %Expr{op: :parameter, id: id, args: _}} = t, acc) do
    {expr, params, var_map, cache} = acc
    {var, var_map} = var_for_id(var_map, id)
    param = %TS{id: var, op: :parameter, args: []}
    {t, {expr, [param | params], var_map, Map.put(cache, id, param)}}
  end

  defp cached_inspect_expr(%T{data: %Expr{op: :tensor, id: id}} = t, acc) do
    {expr, params, var_map, cache} = acc
    {var, var_map} = var_for_id(var_map, id)
    param = %TS{id: var, op: :tensor, args: []}
    {t, {expr, [param | params], var_map, Map.put(cache, id, param)}}
  end

  defp cached_inspect_expr(%T{data: %Expr{id: id, op: op}} = t, acc) do
    {_, {_, params, var_map, cache}} = Tree.apply_args(t, acc, &inspect_expr/2)
    {var, var_map} = var_for_id(var_map, id)

    {args, _} =
      Tree.apply_args(t, nil, fn
        %Nx.Tensor{data: data}, acc -> {cache[data.id], acc}
        arg, acc -> {arg, acc}
      end)

    expr = %TS{id: var, op: op, args: args}
    {t, {expr, params, var_map, Map.put(cache, id, expr)}}
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
