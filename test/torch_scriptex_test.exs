defmodule TorchScriptexTest do
  use ExUnit.Case
  doctest TorchScriptex

  test "greets the world" do
    t = TorchScriptex.hello()
    result = Nx.tensor([[4.0, 6.0], [8.0, 10.0]])
    assert t |> Nx.equal(result) |> Nx.all() |> Nx.to_number() == 1
  end
end
