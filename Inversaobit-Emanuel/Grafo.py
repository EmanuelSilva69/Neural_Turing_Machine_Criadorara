from graphviz import Digraph
from NTM import NTM

def desenhar_estrutura_NTM(model: NTM, nome='ntm_visual', format='png'):
    dot = Digraph(name=nome, format=format)
    dot.attr(rankdir="LR", fontsize="12")

    # Dados do modelo
    input_size = model.input_size
    output_size = model.output_size
    controller_size = model.controller_size
    N = model.N
    M = model.M

    # Nós principais
    dot.node("Entrada", f"Entrada\n({input_size} bits)", shape="box")
    dot.node("Concat", f"Concatenação\n[Entrada + Vetor Lido]", shape="diamond")
    dot.node("Controller", f"Controller\nLinear + ReLU\n({input_size + M} → {controller_size})", shape="ellipse")
    dot.node("WriteHead", "Cabeça de Escrita", shape="parallelogram", style="filled", fillcolor="lightblue")
    dot.node("ReadHead", "Cabeça de Leitura", shape="parallelogram", style="filled", fillcolor="lightgreen")
    dot.node("Memória", f"Memória Externa\n{N}×{M}", shape="cylinder", style="filled", fillcolor="lightgray")
    dot.node("ConcatOut", f"Concatenação\n[Controller + Leitura]", shape="diamond")
    dot.node("Output", f"Saída\nSigmoid\n({controller_size + M} → {output_size})", shape="box")

    # Conexões
    dot.edge("Entrada", "Concat")
    dot.edge("Concat", "Controller")
    dot.edge("Controller", "WriteHead")
    dot.edge("Controller", "ReadHead")
    dot.edge("WriteHead", "Memória", label="Escreve")
    dot.edge("ReadHead", "Memória", label="Lê")
    dot.edge("Memória", "ReadHead")
    dot.edge("ReadHead", "ConcatOut")
    dot.edge("Controller", "ConcatOut")
    dot.edge("ConcatOut", "Output")

    # Renderizar
    caminho = dot.render(filename=nome, cleanup=True)
    print(f"✅ Diagrama salvo em: {caminho}")

if __name__ == "__main__":
    model = NTM(input_size=8, output_size=8)
    desenhar_estrutura_NTM(model)