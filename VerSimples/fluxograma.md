graph TD
    subgraph Geração de Dados
        A[GeradorDataset.py] --> B{copy_dataset.json}
    end

    subgraph Treinamento
        C[train.py] --> B
        C --> D[NTM.py]
        C --> E{checkpoint/best_model.pth}
        C --Gera Gráfico--> F[Gráfico de Perda de Treinamento e Validação]
    end

    subgraph Avaliação Geral
        G[evaluationGeral.py] --Gera Dataset Temporário--> H{dataset_avaliacao.json}
        G --> D
        G --> E
        G --> I{avaliacao_resultados.txt}
        G --> J{avaliacao_exemplos.txt}
    end

    subgraph Avaliação Interativa e Visualização
        K[evaluation.py] --> H
        K --> D
        K --> E
        L[visualizacao.py] --> H
        L --> D
        L --> E
        L --Gera Gráficos--> M[Gráficos de Comparação Visual]
    end

    B[copy_dataset.json] --> C
    B --> K
    B --> L
    D[NTM.py] --Define Arquitetura NTM--> C
    D --Define Arquitetura NTM--> G
    D --Define Arquitetura NTM--> K
    D --Define Arquitetura NTM--> L
    E[checkpoint/best_model.pth] --Pesos do Modelo Treinado--> G
    E --Pesos do Modelo Treinado--> K
    E --Pesos do Modelo Treinado--> L

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#f9f,stroke:#333,stroke-width:2px
    style G fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#f9f,stroke:#333,stroke-width:2px
    style L fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#add8e6,stroke:#333,stroke-width:2px
    style H fill:#add8e6,stroke:#333,stroke-width:2px
    style E fill:#add8e6,stroke:#333,stroke-width:2px
    style F fill:#90EE90,stroke:#333,stroke-width:2px
    style I fill:#90EE90,stroke:#333,stroke-width:2px
    style J fill:#90EE90,stroke:#333,stroke-width:2px
    style M fill:#90EE90,stroke:#333,stroke-width:2px
