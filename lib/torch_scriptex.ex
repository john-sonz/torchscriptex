defmodule TorchScriptex do
  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  import Nx.Defn

  def inspect_torch(var_or_vars, fun) when is_function(fun, 1) do
    jit_apply(
      fn var_or_vars ->
        var_or_vars
        |> fun.()
        |> inspectx2()
      end,
      [var_or_vars],
      on_conflict: :reuse
    )
  end

  defn test_defn(t, p, r) do
    a = p ||| r

    ret =
      Nx.add(t, p)
      |> Nx.subtract(r)
      |> Nx.multiply(2)
      |> Nx.divide(4)
      |> Nx.cbrt()

    {_, ret} =
      while {i = 0, ret}, i < 5 do
        {i + 2, ret + 1}
      end

    (ret + a * Nx.tensor([2, 2]))
    |> Nx.negate()
  end

  defn slice_test() do
    t =
      Nx.tensor([
        [[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12, 13, 14, 15, 16, 17]],
        [[71, 72, 73, 74, 75, 76, 77, 78], [81, 82, 83, 84, 85, 86, 87, 88]]
      ])

    Nx.slice(t, [2, 0, 1], [2, 1, 2], strides: [1, 1, 2])
    |> print_expr()

    # TorchScriptex.inspect_torch({t}, fn {a} -> Nx.slice(a,[2,0,1],[2,1,2],strides: [1,2,3]) end)
    # Nx.Defn.
  end

  def tensor_test(a, b, c) do
    TorchScriptex.inspect_torch({a}, fn {a} -> Nx.Random.key(a) end)

    TorchScriptex.inspect_torch({a, b, c}, fn {a, b, c} ->
      Nx.Random.uniform_split(Nx.Random.key(a), b, c)
    end)
  end

  def test do
    TorchScriptex.inspect_torch({16, 2, 3}, fn {a, b, c} -> test_defn(a, b, c) end)
  end

  deftransform inspectx2(tensor) do
    {_, {funs, params, consts, _}, _} = inspectt(tensor, {[], %{}, %{}, %{}}, 0)

    code = TorchScriptex.Generator.python(funs, params, consts)

    IO.puts(code)

    tensor
  end

  # list of functions
  # parameters
  # constants and tensors
  # map of tensors

  defp inspectt(
         %T{data: %Expr{op: :constant, id: id, args: [number]}} = t,
         {funs, params, consts, var_map} = acc,
         _
       ) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs, params, Map.put(consts, var, number), Map.put(var_map, id, var)}, 0}
    end
  end

  defp inspectt(
         %T{data: %Expr{op: :tensor, id: id, args: [tensor]}} = t,
         {funs, params, consts, var_map} = acc,
         _
       ) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs, params, Map.put(consts, var, tensor), Map.put(var_map, id, var)}, 0}
    end
  end

  defp inspectt(
         %T{data: %Expr{op: :parameter, context: :root, id: id, args: [num]}} = t,
         {funs, params, consts, var_map} = acc,
         _
       ) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs, Map.put(params, var, num), consts, Map.put(var_map, id, var)}, 0}
    end
  end

  defp inspectt(
         %T{data: %Expr{op: :parameter, context: context, id: id, args: [num]}} = t,
         {funs, params, consts, var_map} = acc,
         _
       ) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs, params, consts, Map.put(var_map, id, var)}, 0}
    end
  end

  defp inspectt(
         %T{data: %Expr{op: :elem, context: context, id: id, args: [tuple, num]}} = t,
         {funs, params, consts, var_map} = acc,
         depth
       ) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        {tuple, {funs, params, consts, var_map}, _} = inspectt(tuple, acc, depth)
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))

        {var,
         {[{depth, var, :elem, [tuple, num]} | funs], params, consts, Map.put(var_map, id, var)},
         0}
    end
  end

  defp inspectt(
         %T{data: %Expr{op: :while, id: id, args: [init, state, condition, output]}} = t,
         {funs, params, consts, var_map} = acc,
         depth
       ) do
    case var_map do
      %{^id => var} ->
        {var, acc, depth}

      %{} ->
        {init_strings, {funs, params, consts, var_map} = acc} =
          Enum.reduce(
            init |> Tuple.to_list(),
            {[], acc},
            fn arg, {args_string, acc} ->
              {arg_string, acc, _} = inspectt(arg, acc, depth)
              {[arg_string | args_string], acc}
            end
          )

        {state_strings, {funs, params, consts, var_map} = acc} =
          Enum.reduce(
            state |> Tuple.to_list(),
            {[], acc},
            fn arg, {args_string, acc} ->
              {arg_string, acc, _} = inspectt(arg, acc, depth)
              {[arg_string | args_string], acc}
            end
          )

        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))

        acc = {
          [{depth, var, :while, [init_strings, state_strings]} | funs],
          params,
          consts,
          Map.put(var_map, id, var)
        }

        {condition_string, acc, _} = inspectt(condition, acc, depth + 1)

        {out_strings, {funs, params, consts, var_map} = acc} =
          Enum.reduce(
            output |> Tuple.to_list(),
            {[], acc},
            fn arg, {args_string, acc} ->
              {arg_string, acc, _} = inspectt(arg, acc, depth + 1)
              {[arg_string | args_string], acc}
            end
          )

        # {var, {
        #   [{depth, var, :while, [init_strings, state_strings, condition_string, out_strings]} | funs],
        #   params,
        #   consts,
        #   Map.put(var_map, id, var)},
        #   depth}
        {var, acc, depth}
    end
  end

  defp inspectt(
         %T{data: %Expr{op: op, id: id, args: args}} = t,
         {funs, params, consts, var_map} = acc,
         depth
       ) do
    case var_map do
      %{^id => var} ->
        {var, {funs, params, consts, var_map}, depth}

      %{} ->
        {args_strings, {funs, params, consts, var_map} = acc} =
          Enum.reduce(
            args,
            {[], acc},
            fn arg, {args_string, acc} ->
              {arg_string, acc, _} = inspectt(arg, acc, depth)
              {[arg_string | args_string], acc}
            end
          )

        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))

        {var,
         {[{depth, var, op, args_strings} | funs], params, consts, Map.put(var_map, id, var)},
         depth}
    end
  end

  defp counter_to_name(counter) when counter >= 26 do
    [counter_to_name(div(counter, 26)) | counter_to_name(rem(counter, 26))]
  end

  defp counter_to_name(counter), do: [Enum.at(?a..?z, counter)]
end
