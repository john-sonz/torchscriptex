defmodule TorchScriptex do
  alias Nx.Defn.{Composite, Expr, Tree}
  alias Nx.Tensor, as: T

  @enforce_keys [:id, :op, :args, :context]
  defstruct [:id, :op, :args, :context]

  import Nx.Defn

    @unary_ops [:exp, :expm1, :log, :log1p, :sigmoid, :cos, :sin, :tan, :cosh, :sinh, :tanh] ++
      [:acosh, :asinh, :atanh, :sqrt, :rsqrt, :sign, :abs, :bitwise_not] ++
      [:floor, :ceil, :round] ++
      [:erf, :erfc, :acos, :asin, :atan, :real, :imag]

  @unary_exs [:negate, :is_nan, :is_infinity, :erf_inv]
  @unary_exs_replacements %{
        negate: "neg",
        is_nan: "isnan",
        is_infinity: "isinf",
        erf_inv: "erfinv"
      }

  @unary_custom [:cbrt, :conjugate, :count_leading_zeros, :bitcast, :population_count]

  @binary_ops [:remainder, :atan2, :max, :min] ++
      [:bitwise_and, :bitwise_or, :bitwise_xor] ++
      [:equal, :not_equal, :greater, :less, :less_equal, :greater_equal] ++
      [:logical_and, :logical_or, :logical_xor] ++
      [:add, :subtract, :multiply, :divide]

  @binary_exs [:left_shift, :right_shift]
  @binary_exs_replacements %{
    left_shift: "bitwise_left_shift",
    right_shift: "bitwise_right_shift"
  }

  @binary_custom [:quotient]

  def inspect_torch(var_or_vars, fun) when is_function(fun, 1) do
    jit_apply(
      fn var_or_vars ->
        var_or_vars |> fun.()
        |> inspectx2()
      end,
      [var_or_vars],
      on_conflict: :reuse
    )
  end

  defn softmax(t, p, r) do
    a = p ||| r
    ret = Nx.add(t, p)
    |> Nx.subtract(r)
    |> Nx.multiply(2)
    |> Nx.divide(4)

    {_, ret} =
      while {i = 0, ret}, i < 5 do
        {i+2, ret+1}
      end

    ret = ret + a * Nx.tensor([2,2])

    #print_expr(ret, structs: false)
  end

  defn slice_test() do
    t = Nx.tensor([[[1, 2, 3, 4, 5, 6, 7, 8],
    [10, 11, 12, 13, 14, 15, 16, 17]],
   [[71, 72, 73, 74, 75, 76, 77, 78],
    [81, 82, 83, 84, 85, 86, 87, 88]]])

    Nx.slice(t,[2,0,1],[2,1,2],strides: [1,1,2])
    |> print_expr()
    |> inspectx()

    #TorchScriptex.inspect_torch({t}, fn {a} -> Nx.slice(a,[2,0,1],[2,1,2],strides: [1,2,3]) end)
    #Nx.Defn.
  end

  def tensor_test(a, b, c) do
    #Nx.LinAlg.cholesky(Nx.tensor([[20.0, 17.6], [17.6, 16.0]]))
    #TorchScriptex.inspect_torch(Nx.tensor([[20.0, 17.6], [17.6, 16.0]]), fn x -> Nx.LinAlg.cholesky(x) end)
    TorchScriptex.inspect_torch({a}, fn {a} -> Nx.Random.key(a) end)
    TorchScriptex.inspect_torch({a,b,c}, fn {a,b,c} -> Nx.Random.uniform_split(Nx.Random.key(a),b,c) end)
  end

  defn test do
    Nx.Random.uniform_split(Nx.Random.key(1701),2,3)
    |> print_expr(structs: false)
  end

  def test_softmax do
    TorchScriptex.inspect_torch({16, 2, 3}, fn {a, b, c} -> softmax(a,b,c) end)
  end

  deftransform inspectx2(tensor) do
    {t, acc, _} = inspectt(tensor, {[], %{}, %{}, %{}}, 0)
    IO.inspect(acc)
    tensor
  end

  deftransform inspectx(tensor) do
    {t, acc} = inspect_expr(tensor, {[], [], %{}, %{}})

    {_, {exprs, params, var_map, cache}} = Tree.apply_args(t, acc, &inspect_expr/2)
    # IO.inspect(var_map)
    # IO.puts("====================================")
    # IO.inspect(cache)

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
    File.write("test.py", all)
    tensor
  end

  # list of functions
  # parameters
  # constants and tensors
  # map of tensors

  #defp inspect(%T{data: %Expr{op: :constant}}, acc)
  defp inspectt(%T{data: %Expr{op: :constant, id: id, args: [number]}} = t, {funs, params, consts, var_map} = acc, _) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs,
        params,
        Map.put(consts, var, number),
        Map.put(var_map, id, var)},
        0}
    end
  end

  defp inspectt(%T{data: %Expr{op: :tensor, id: id, args: [tensor]}} = t, {funs, params, consts, var_map} = acc, _) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs,
        params,
        Map.put(consts, var, tensor),
        Map.put(var_map, id, var)},
        0}
    end
  end

  defp inspectt(%T{data: %Expr{op: :parameter, context: :root, id: id, args: [num]}} = t,
  {funs, params, consts, var_map} = acc,
  _) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs,
        Map.put(params, var, num),
        consts,
        Map.put(var_map, id, var)},
        0}
    end
  end

  defp inspectt(%T{data: %Expr{op: :parameter, context: context, id: id, args: [num]}} = t,
  {funs, params, consts, var_map} = acc,
  _) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {funs,
        params,
        consts,
        Map.put(var_map, id, var)},
        0}
    end
  end

  defp inspectt(%T{data: %Expr{op: :elem, context: context, id: id, args: [tuple, num]}} = t,
  {funs, params, consts, var_map} = acc,
  depth) do
    case var_map do
      %{^id => var} ->
        {var, acc, 0}

      %{} ->
        {tuple, {funs, params, consts, var_map}, _} = inspectt(tuple, acc, depth)
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {[ {depth, var, :elem, [tuple, num]} | funs],
        params,
        consts,
        Map.put(var_map, id, var)},
        0}
    end
  end

  defp inspectt(%T{data: %Expr{op: :while, id: id, args: [init, state, condition, output]}} = t,
  {funs, params, consts, var_map} = acc,
  depth) do
    case var_map do
      %{^id => var} ->
        {var, acc, depth}

      %{} ->

        {init_strings, {funs, params, consts, var_map} = acc} = Enum.reduce(
          init |> Tuple.to_list,
          {[], acc},
          fn arg, {args_string, acc} ->
          {arg_string, acc, _} = inspectt(arg, acc, depth)
          {[arg_string | args_string], acc}
        end)

        {state_strings, {funs, params, consts, var_map} = acc} = Enum.reduce(
          state |> Tuple.to_list,
          {[], acc},
          fn arg, {args_string, acc} ->
          {arg_string, acc, _} = inspectt(arg, acc, depth)
          {[arg_string | args_string], acc}
        end)

        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        acc = {
          [{depth, var, :while, [init_strings, state_strings]} | funs],
          params,
          consts,
          Map.put(var_map, id, var)}

        {condition_string, acc, _} = inspectt(condition, acc, depth+1)

        {out_strings, {funs, params, consts, var_map} = acc} = Enum.reduce(
          output |> Tuple.to_list,
          {[], acc},
          fn arg, {args_string, acc} ->
          {arg_string, acc, _} = inspectt(arg, acc, depth+1)
          {[arg_string | args_string], acc}
        end)


        # {var, {
        #   [{depth, var, :while, [init_strings, state_strings, condition_string, out_strings]} | funs],
        #   params,
        #   consts,
        #   Map.put(var_map, id, var)},
        #   depth}
        {var, acc, depth}
    end
  end

  defp inspectt(%T{data: %Expr{op: op, id: id, args: args}} = t, {funs, params, consts, var_map} = acc, depth) do
    IO.inspect(op)
    case var_map do
      %{^id => var} ->
        {var, {funs, params, consts, var_map}, depth}

      %{} ->
        {args_strings, {funs, params, consts, var_map} = acc} = Enum.reduce(
          args,
          {[], acc},
          fn arg, {args_string, acc} ->
          {arg_string, acc, _} = inspectt(arg, acc, depth)
          {[arg_string | args_string], acc}
        end)
        var = IO.iodata_to_binary(counter_to_name(map_size(var_map)))
        {var, {[ {depth, var, op, args_strings} | funs],
        params,
        consts,
        Map.put(var_map, id, var)},
        depth}
      end
  end

  # Constants and funs are shown as is
  defp inspect_expr(%T{data: %Expr{op: :constant}} = t, acc), do: {t, acc}
  defp inspect_expr(%T{data: %Expr{op: :fun}} = t, acc), do: {t, acc} |> IO.inspect()

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

  defp inspect_nested_expr(%T{data: %Expr{id: id}} = t, {exprs, params, var_map, cache} = acc) do
    case cache do
      %{^id => _} -> {t, acc}
      %{} -> cached_inspect_nested_expr(t, {exprs, params, var_map, Map.put(cache, id, true)})
    end
  end


  defp cached_inspect_nested_expr(%T{data: %Expr{id: id, op: op}} = t, acc) do
    {args, {exprs, params, var_map, cache}} = traverse_args(op, t, acc)
    {var, var_map} = var_for_id(var_map, id)
    args_str = inspect_args(op, args, var_map)
    expr_str = var <> " = " <> pythonize(t, op, args_str)
    # IO.inspect([{expr_str} | exprs])
    # IO.inspect(params)
    # IO.inspect("===========================")
    {t, {[{expr_str} | exprs], params, var_map, cache}}
  end

  defp cached_inspect_expr(%T{data: %Expr{op: :parameter, id: id, args: [i]}} = t, acc) do
    {exprs, params, var_map, cache} = acc
    {var, var_map} = var_for_id(var_map, id)
    param = {"parameter", var <> "=" <> Integer.to_string(i)}
    #IO.inspect(var_map)
    {t, {exprs, [param | params], var_map, cache}}
  end

  defp cached_inspect_expr(%T{data: %Expr{op: :tensor, id: id}} = t, acc) do
    {exprs, params, var_map, cache} = acc
    {var, var_map} = var_for_id(var_map, id)
    param = {"tensor ", var}
    {t, {exprs, [param | params], var_map, cache}}
  end

  defp cached_inspect_expr(%T{data: %Expr{id: id, op: :while = op}} = t, acc) do
    {args, {exprs, params, var_map, cache}} = traverse_args(op, t, acc)
    {var, var_map} = var_for_id(var_map, id)
    args_str = inspect_args(op, args, var_map)
    expr_str = var <> " = " <> pythonize(t, op, args_str)
    {t, {[{expr_str} | exprs], params, var_map, cache}}
  end

  # defp cached_inspect_expr(%T{data: %Expr{id: id, op: :while = op}} = t, acc) do
  #   {args, {exprs, params, var_map, cache}} = traverse_args_while_initial(t, acc)
  #   {var, var_map} = var_for_id(var_map, id)

  #   IO.inspect({var, args})



  #   {args, {exprs, params, var_map, cache}} = traverse_args_while_block(t, {exprs, params, var_map,cache})
  #   {var_2, var_map} = var_for_id(var_map, id)

  #   IO.inspect({var_2, args})

  #   args_str = inspect_args(op, args, var_map)
  #   expr_str = var <> " = " <> pythonize(t, op, args_str)
  #   {t, {[{expr_str} | exprs], params, var_map, cache}}
  # end

  defp cached_inspect_expr(%T{data: %Expr{id: id, op: op}} = t, acc) do
    {args, {exprs, params, var_map, cache}} = traverse_args(op, t, acc)
    {var, var_map} = var_for_id(var_map, id)
    args_str = inspect_args(op, args, var_map)
    expr_str = var <> " = " <> pythonize(t, op, args_str)
    # IO.inspect([{expr_str} | exprs])
    # IO.inspect(params)
    # IO.inspect("===========================")
    {t, {[{expr_str} | exprs], params, var_map, cache}}
  end

  defp pythonize(_t, op, args_str) when op in @unary_ops do
    "torch." <> Atom.to_string(op) <> "(" <> args_str <> ")"
  end

  defp pythonize(_t, op, args_str) when op in @unary_exs do
    "torch." <> Map.get(@unary_exs_replacements, op) <> "(" <> args_str <> ")"
  end

  defp pythonize(_t, op, args_str) when op in @binary_ops do
    "torch." <> Atom.to_string(op) <> "(" <> args_str <> ")"
  end

  defp pythonize(_t, op, args_str) when op in @binary_exs do
    "torch." <> Map.get(@binary_exs_replacements, op) <> "(" <> args_str <> ")"
  end

  defp pythonize(t, op, args_str) do
    args_str = args_str
    |> String.split(",")
    pythonize_custom(t, op, args_str)
  end

  defp pythonize_custom(%T{type: type} = _t, :as_type, [tensor]) do
    tensor <> ".to(" <> python_type(type) <> ")"
  end

  defp pythonize_custom(%T{shape: shape} = _t, :reshape, [tensor]) do
    tensor <> ".reshape(" <> tensor <> ", " <> python_list(shape) <> ")"
  end

  defp pythonize_custom(%T{shape: shape} = _t, :slice, [tensor, start, lengths, strides]) do
    start = unwrap_list(start)
    lengths = unwrap_list(lengths)
    strides = unwrap_list(strides)
    dim = length(start)

    narrowed =
      Enum.zip([start, lengths, 0..dim-1])
      |> Enum.reduce(tensor, fn ({s, l, i}, acc) -> "torch.narrow(#{acc}, #{i}, #{s}, #{l})" end)

    output_shape = python_list(shape)

    if Enum.all?(strides, &(&1 == "1")) do
      narrowed
    else
      strides =
        Enum.map(lengths, &(String.to_integer(&1)))
        |> steps_to_strides(Enum.map(strides, &(String.to_integer(&1))))
        |> python_list()

      "torch.as_strided(#{narrowed}.contiguous(), #{output_shape}, #{strides}, 0)"
    end

  end

  def steps_to_strides(shape, steps) do
    for {dim, step} <- Enum.zip(shape, steps) |> Enum.reverse(), reduce: {1, []} do
      {offset, strides} -> {offset * dim, [offset * step | strides]}
    end
    |> elem(1)
  end

  defp pythonize_custom(_t, :squeeze, [tensor, axes]) do
    axes = unwrap_list(axes)

    for axis <- axes, reduce: tensor do
      acc -> "torch.squeeze(#{acc}, #{axis})"
    end
  end

  defp pythonize_custom(_t, :concatenate, [tensors, axis]) do
    tensors =
      tensors
      |> unwrap_list()
      |> wrap_list()

    "torch.cat(#{tensors}, #{axis})"
  end

  # defp pythonize_custom(_t, :while, _) do
  #   "while: \n\tindented line 1" <> "\n\t\tindented line 2"
  # end

  defp pythonize_custom(_t, op, args_str) do
    Atom.to_string(op) <> " " <> Enum.join(args_str, " ")
  end

  defp traverse_args(:while, %T{data: %{args: [initial, params, condition, output]} } = t, acc) do
    # IO.inspect("================INITIAL==================")
    # IO.inspect(initial)
    # IO.inspect("================PARAMS==================")
    # IO.inspect(params)
    # IO.inspect("================CONDITION==================")
    # IO.inspect(condition)
    # IO.inspect("================OUTPUT==================")
    # IO.inspect(output)

    #IO.inspect(output)

    {initial, acc} = Composite.traverse(initial, acc, &inspect_expr/2)
    #{params, acc} = Composite.traverse(params, acc, &inspect_expr/2)
    {output, acc} = Composite.traverse(output, acc, &inspect_nested_expr/2)
    {condition, acc} = Composite.traverse(condition, acc, &inspect_expr/2)




    # {initial, acc} = traverse_args_while_initial(t, acc)
    # IO.inspect(initial)
    # {condition, acc} = traverse_args_while_condition(initial, acc)
    # {output, acc} = traverse_args_while_output(condition, acc)

    {[output], acc}
  end

  defp traverse_args_while_initial(%T{data: %{args: [initial, params, condition, block]} } , acc) do
    {initial, acc} = Composite.traverse(initial, acc, &inspect_expr/2)
    {[initial], acc}
  end

  defp traverse_args_while_condition(%T{data: %{args: [initial, params, condition, block]} }, acc) do
    {condition, acc} = Composite.traverse(initial, acc, &inspect_expr/2)
    {[condition], acc}
  end

  defp traverse_args_while_block(%T{data: %{args: [initial, params, condition, block]} }, acc) do
    {block, acc} = Composite.traverse(condition, acc, &inspect_expr/2)
    {[block], acc}
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

  defp traverse_nested_args(_op, t, acc) do
    Tree.apply_args(t, acc, &inspect_nested_expr/2)
  end

  defp inspect_args(:token, hooks, var_map) do
    IO.iodata_to_binary(
      Enum.map_intersperse(hooks, ", ", fn {key, val} ->
        "#{key}: " <> inspect_arg(val, var_map)
      end)
    )
  end

  defp inspect_args(:while, [initial], var_map) do
    # IO.puts("====================================")
    # IO.inspect(Tuple.to_list(initial))
    # IO.puts("====================================")
    {exprs, var_map} = Tuple.to_list(initial)
    |> Enum.reduce({[], var_map}, fn (tensor, {list, vars}) ->
      {t, acc} = inspect_expr(tensor, {[], [], vars, %{}})
      {_, {exprs, params, var_map, _cache}} = Tree.apply_args(t, acc, &inspect_expr/2)
      {[exprs | list], var_map}
    end)

    #IO.inspect(exprs)

    #IO.inspect(inspect_expr(initial, {[], [], var_map, %{}}))
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
    do: Enum.map_join(args, ",", &inspect_arg(&1, var_map))

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
            [?[, inspect_args(arg, var_map) |> String.replace(",", "|"), ?]]

          is_tuple(arg) ->
            [?{, inspect_args(Tuple.to_list(arg), var_map) |> String.replace(",", "&"), ?}]

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

  defp python_list(tuple) when is_tuple(tuple) do
    python_list(Tuple.to_list(tuple))
  end

  defp python_list(list) when is_list(list) do
    "#{inspect list}"
  end

  defp unwrap_tuple(tuple) do
    tuple
    |> String.trim_leading("(")
    |> String.trim_trailing(")")
    |> String.split("&")
  end

  defp unwrap_list(list) do
    list
    |> String.trim_leading("[")
    |> String.trim_trailing("]")
    |> String.split("|")
  end

  defp wrap_list(list) do
    "[" <> Enum.join(list, ", ") <> "]"
  end
end
