defmodule TorchScriptex do
  def hello do
    Nx.default_backend(Torchx.Backend)

    Nx.iota({2, 2})
    |> Nx.add(2.0)
    |> Nx.multiply(2)
  end
end
