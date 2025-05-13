import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    filename='simplex.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Alterar o nome da aba para 'Modelo_2', 'Modelo_3', para usar outro modelo
# df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_1')
# df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_2')
# df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_3')
# df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_4')
# df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_5')
# df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_6')
df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_7')
# df = pd.read_excel('Modelos_Simplex.xlsx', sheet_name='Modelo_8')

print("\n====== LENDO O MODELO ======")
print(df)
logging.info("Lendo o modelo.")

# Variáveis e parâmetros do modelo
tipo_objetivo = df.iloc[0, 0]

variaveis = df.columns[1:-2].tolist()
variaveis = np.array(variaveis).reshape(-1, 1)

c = df.iloc[0, 1:-2].tolist()
c = np.array(c).reshape(1, -1)

A = df.iloc[1:, 1:-2].fillna(0).values

b = df.iloc[1:, -1].values.reshape(-1, 1)

sinais = df.iloc[1:, -2].tolist()

# Exibindo o modelo inicial
print("\n====== MODELO PRIMAL ======")
print("\nModelo primal:")
print(f"- Tipo: {tipo_objetivo}")
print(f"- Vetor x: {variaveis}")
print(f"- Vetor c: {c}")
print(f"- Matriz A:\n{A}")
print(f"- Vetor b: {b}")
print(f"- Sinais das restricoes: {sinais}")

logging.info("Imprimindo modelo primal.")
logging.debug(f"- Tipo: {tipo_objetivo}")
logging.debug(f"- Vetor x: {variaveis}")
logging.debug(f"- Vetor c: {c}")
logging.debug(f"- Matriz A:\n{A}")
logging.debug(f"- Vetor b: {b}")
logging.debug(f"- Sinais das restrições: {sinais}")

# Transformar o modelo primal para a forma padrão
if tipo_objetivo == 'Max':
    tipo_objetivo_padrao = 'Min'
    c_padrao = -c
else:
    tipo_objetivo_padrao = 'Min'
    c_padrao = c

novas_variaveis = variaveis.flatten().tolist()
A_padrao = A.copy()
sinais_padrao = []
c_padrao = c_padrao.flatten().tolist()

# Processando as restrições para a forma padrão
for i, sinal in enumerate(sinais):
    if sinal == '<=':
        novas_variaveis.append(f'f{len(novas_variaveis) + 1}')
        nova_coluna = np.zeros(A_padrao.shape[0])
        nova_coluna[i] = 1
        A_padrao = np.column_stack((A_padrao, nova_coluna))
        sinais_padrao.append('=')
        c_padrao.append(0)
    elif sinal == '>=':
        novas_variaveis.append(f'e{len(novas_variaveis) + 1}')
        nova_coluna = np.zeros(A_padrao.shape[0])
        nova_coluna[i] = -1
        A_padrao = np.column_stack((A_padrao, nova_coluna))
        sinais_padrao.append('=')
        c_padrao.append(0)
        novas_variaveis.append(f'w{len(novas_variaveis) + 1}')
        nova_coluna = np.zeros(A_padrao.shape[0])
        nova_coluna[i] = 1
        A_padrao = np.column_stack((A_padrao, nova_coluna))
        c_padrao.append(0)
    else:
        novas_variaveis.append(f'w{len(novas_variaveis) + 1}')
        nova_coluna = np.zeros(A_padrao.shape[0])
        nova_coluna[i] = 1
        A_padrao = np.column_stack((A_padrao, nova_coluna))
        c_padrao.append(0)
        sinais_padrao.append('=')

novas_variaveis = np.array(novas_variaveis).reshape(-1, 1)
c_padrao = np.array(c_padrao).reshape(1, -1)

print("\n====== MODELO PRIMAL NA FORMA PADRÃO ======")
print("\nModelo primal na forma padrão:")
print(f"- Tipo: {tipo_objetivo_padrao}")
print(f"- Vetor x: {novas_variaveis}")
print(f"- Vetor c: {c_padrao}")
print(f"- Matriz A:\n{A_padrao}")
print(f"- Vetor b: {b}")
print(f"- Sinais das restrições: {sinais_padrao}")

logging.info("Transformando modelo primal na forma padrão.")
logging.debug(f"- Tipo: {tipo_objetivo_padrao}")
logging.debug(f"- Vetor x: {novas_variaveis}")
logging.debug(f"- Vetor c: {c_padrao}")
logging.debug(f"- Matriz A:\n{A_padrao}")
logging.debug(f"- Vetor b: {b}")
logging.debug(f"- Sinais das restrições: {sinais_padrao}")

# Gerando o problema dual
print("\n====== GERAÇÃO DO PROBLEMA DUAL ======")
logging.info("Gerando modelo dual.")
if tipo_objetivo == 'Max':
    tipo_objetivo_dual = 'Min'
else:
    tipo_objetivo_dual = 'Max'

c_dual = b.flatten()
c_dual = np.array(c_dual).reshape(1, -1)

A_dual = A.T

b_dual = c.flatten()
b_dual = np.array(b_dual).reshape(-1, 1)

sinais_restricoes_dual = []
for s in sinais:
    if s == '>=':
        sinais_restricoes_dual.append('>=')
    elif s == '<=':
        sinais_restricoes_dual.append('<=')
    else:
        sinais_restricoes_dual.append('=')


variaveis_dual = [f'y{i+1}' for i in range(len(c_dual.flatten()))]
variaveis_dual = np.array(variaveis_dual).reshape(-1, 1)

sinais_variaveis_dual = []
for i, s in enumerate(sinais):
    if s == '<=':
        sinais_variaveis_dual.append(f'{variaveis_dual[i, 0]} >= 0')
    elif s == '>=':
        sinais_variaveis_dual.append(f'{variaveis_dual[i, 0]} <= 0')
    else:  # s == '='
        sinais_variaveis_dual.append(f'{variaveis_dual[i, 0]} livre')

# Exibindo o modelo dual
print("\nModelo dual:")
print(f"- Tipo: {tipo_objetivo_dual}")
print(f"- Vetor y: {variaveis_dual}")
print(f"- Vetor c_dual: {c_dual}")
print(f"- Matriz A_dual:\n{A_dual}")
print(f"- Vetor b_dual: {b_dual}")
print(f"- Sinais das restrições: {sinais_restricoes_dual}")
print(f"- Variaveis: {sinais_variaveis_dual}")

logging.debug(f"- Tipo: {tipo_objetivo_dual}")
logging.debug(f"- Vetor y: {variaveis_dual}")
logging.debug(f"- Vetor b_dual: {b_dual}")
logging.debug(f"- Matriz A_dual:\n{A_dual}")
logging.debug(f"- Vetor c_dual: {c_dual}")
logging.debug(f"- Sinais das restrições: {sinais_restricoes_dual}")
logging.debug(f"- Variaveis: {sinais_variaveis_dual}")

# Identificando as variáveis básicas e não básicas
m, n = A_padrao.shape
indices_base = []

for j in range(n):
    coluna = A_padrao[:, j]
    identidade_coluna = np.zeros(m)
    for i in range(m):
        identidade_coluna[i] = 1
        if np.array_equal(coluna, identidade_coluna) or np.array_equal(coluna, -identidade_coluna):
            indices_base.append(j)
            break
        identidade_coluna[i] = 0

indices_nao_base = [j for j in range(n) if j not in indices_base]
indices_artificiais = [
    j for j, nome in enumerate(novas_variaveis.flatten()) if 'w' in str(nome)
]

# Metodo 2 fases
print("\n====== VERIFICANDO SE O PROBLEMA É FACTIVEL ======")
logging.info("Verificando de o modelo e factivel.")

tem_artificiais = False
for i, sinal in enumerate(sinais):
    if sinal in ('=', '>='):
        print(f"Restrição {i+1}: sinal '{sinal}' → requer variável artificial.")
        tem_artificiais = True

if tem_artificiais:
    print("\nO problema tem variáveis artificiais.")
    logging.info("Modelo tem variaveis artificiais.")

    c_padrao_2_fase = c_padrao[0].copy()
    c_padrao_2_fase[indices_base] = 0
    c_padrao_2_fase[indices_nao_base] = 0
    c_padrao_2_fase[indices_artificiais] = 1

    tamanho_total_variaveis = len(c_padrao_2_fase)

    indices_artificiais_ordem = [i for i, nome in enumerate(novas_variaveis.flatten()) if str(nome).startswith('w')]
    indices_nao_artificiais_ordem = [i for i in range(len(novas_variaveis)) if i not in indices_artificiais]
    indices_artificiais = [i for i, nome in enumerate(novas_variaveis.flatten()) if str(nome).startswith('f') or str(nome).startswith('w')]
    indices_nao_artificiais = [i for i in range(len(novas_variaveis)) if i not in indices_artificiais]
    ordem = indices_nao_artificiais_ordem + indices_artificiais_ordem
    nova_ordem = ordem.copy()

    c_padrao_2_fase = c_padrao_2_fase.reshape(1, -1)
    c_padrao_2_fase = c_padrao_2_fase[:, nova_ordem]

    indices_artificiais_novos = [nova_ordem.index(i) for i in indices_artificiais]
    indices_nao_artificiais_novos = [nova_ordem.index(i) for i in indices_nao_artificiais]

    print(f"Nova função objetivo (Fase 1): {c_padrao_2_fase}")
    logging.debug(f"Nova função objetivo (Fase 1): {c_padrao_2_fase}")

    novas_variaveis_reordenadas = novas_variaveis.flatten()
    novas_variaveis_reordenadas = novas_variaveis_reordenadas[nova_ordem].reshape(-1, 1)

    A_reordenada = A_padrao[:, nova_ordem]

    print(f"\n===== SIMPLEX - FASE 1 =====")
    logging.info("Resolvendo o simplex fase 1.")
    print("\nResolvendo o problema:")
    try:
        B = A_padrao[:, indices_artificiais].astype(np.float64)
        print(f'\n - Matriz B (base):\n{B}')
        logging.debug("Invertendo matriz A.")
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        print("Erro: a matriz B não é invertível.")
        logging.error("A matriz B não é invertível.")
        B_inv = None

    if B_inv is not None:
        N = A_padrao[:, indices_nao_artificiais]
        print(f'- Matriz N (não-base):\n{N}')

        # Cálculo da solução básica viável
        print("\n - Resolvendo o sistema para encontrar solução básica viável")
        logging.info("Resolvendo o sistema para encontrar solução básica viável")
        x_nao_artificial = np.zeros(len(indices_nao_artificiais))
        x_artificial = np.dot(B_inv, b)

        p = np.dot(c_padrao_2_fase[:, indices_artificiais_novos], B_inv)
        z = np.dot(p, b) + sum((c_padrao_2_fase[:, indices_nao_artificiais_novos] - np.dot(p, N[:, j])) * x_nao_artificial[j] for j in range(len(indices_nao_artificiais)))

        solucao_geral = np.dot(B_inv, (np.dot(N, x_nao_artificial) + np.dot(B, x_artificial) - b))

        print(f'Valor da função objetivo: {np.dot(z[0][0],-1)}')
        print("Solução básica viável:")
        for i, var in enumerate(novas_variaveis_reordenadas):
            if i in indices_artificiais_novos:
                print(f"{var[0]} = {x_artificial[indices_artificiais_novos.index(i)][0]}")
            else:
                print(f"{var[0]} = {x_nao_artificial[indices_nao_artificiais_novos.index(i)]}")
        print(f"Resultado solução geral:\n{solucao_geral}")

        logging.debug(f'Valor da função objetivo: {np.dot(z[0][0],-1)}')
        logging.info("Solução básica viável:")
        for i, var in enumerate(novas_variaveis_reordenadas):
            if i in indices_artificiais_novos:
                logging.debug(f"{var[0]} = {x_artificial[indices_artificiais_novos.index(i)][0]}")
            else:
                logging.debug(f"{var[0]} = {x_nao_artificial[indices_nao_artificiais_novos.index(i)]}")

        # Tem como melhorar a solução?
        iteracao = 0
        max_iteracoes = 10
        while iteracao < max_iteracoes:
            iteracao += 1
            print(f"\n=== ITERAÇÃO {iteracao} ===")
            logging.info(f"Interação {iteracao}")
            print("\n - Tem como melhorar a solução básica?")

            s_j = [c_padrao_2_fase[0][j] - np.dot(p, N[:, i]) for i, j in enumerate(indices_nao_artificiais_novos)]
            print("Custos reduzidos:")
            logging.info("Custos Reduzidos:")
            for i, j in enumerate(indices_nao_artificiais_novos):
                print(f"s{j + 1} = {s_j[i]}")
                logging.debug(f"s{j + 1} = {s_j[i]}")

            epsilon = 1e-8

            menor_c = np.min(s_j)
            print(f"Menor valor em c: {menor_c}")
            logging.debug(f"Menor valor em c: {menor_c}")
            if menor_c < -epsilon:
                print("Sim, é possível melhorar a solução.")
                logging.debug("Sim, é possível melhorar a solução.")
            else:
                print("Não, não é possível melhorar a solução.")
                logging.debug(("Não, não é possível melhorar a solução."))
                print("\n ===== SOLUÇÃO ÓTIMA ENCONTRADA ======")
                logging.info("Solução otima encontrada:")
                print(f"\nSolução ótima encontrada na iteração {iteracao}:")
                print(f"Valor ótimo da função objetivo: {np.dot(z[0][0], -1)}")
                logging.debug(f"Valor ótimo da função objetivo: {np.dot(z[0][0], -1)}")
                print("Solução ótima:")
                logging.info("Solução ótima:")

                ultimos_indices_artificiais = []
                for i in indices_artificiais_novos:
                    ultimos_indices_artificiais.append(i)

                for i, var in enumerate(novas_variaveis_reordenadas):
                    if i in indices_artificiais_novos:
                        print(f"{var[0]} = {x_artificial[indices_artificiais_novos.index(i)][0]}")
                        logging.debug(f"{var[0]} = {x_artificial[indices_artificiais_novos.index(i)][0]}")
                    else:
                        print(f"{var[0]} = {x_nao_artificial[indices_nao_artificiais_novos.index(i)]}")
                        logging.debug(f"{var[0]} = {x_nao_artificial[indices_nao_artificiais_novos.index(i)]}")
                break

            indice_entrada_nao_base = np.argmin(s_j)
            indice_entrada = indices_nao_artificiais_novos[indice_entrada_nao_base]

            if menor_c >= -epsilon:
                print("Não há mais variáveis que podem entrar na base.")
                logging.info("Não há mais variáveis que podem entrar na base.")
            var_entrada = novas_variaveis_reordenadas[indice_entrada]
            print(f"\nVariável que vai entrar na base: {var_entrada}")
            logging.debug(f"\nVariável que vai entrar na base: {var_entrada}")

            y = np.dot(B_inv, A_padrao[:, indice_entrada])

            razoes = []
            for i in range(len(indices_artificiais_novos)):
                x_base_val = x_artificial[i][0]
                if y[i] > 0:
                    razoes.append(x_base_val / y[i])
                else:
                    razoes.append(np.inf)

            min_razao = min(razoes)
            if min_razao == np.inf:
                print("Não há solução viável, pois não há limite superior.")
                logging.error("Não há solução viável, pois não há limite superior.")
                break

            indice_saida_base = np.argmin(razoes)
            indice_saida = indices_artificiais_novos[indice_saida_base]
            var_saida = novas_variaveis_reordenadas[indice_saida]
            print(f"Variável que vai sair da base: {var_saida}")
            logging.debug(f"Variável que vai sair da base: {var_saida}")
            print(f"Valor máximo que a variável de entrada pode assumir: {min_razao}")
            logging.debug(f"Valor máximo que a variável de entrada pode assumir: {min_razao}")

            indices_artificiais_novos[indice_saida_base] = indice_entrada
            indices_nao_artificiais_novos[indice_entrada_nao_base] = indice_saida

            B = A_reordenada[:,indices_artificiais_novos].astype(np.float64)
            B_inv = np.linalg.inv(B)
            N = A_reordenada[:,indices_nao_artificiais_novos]
            print(f'\n - Matriz B (base):\n{B}')
            print(f'- Matriz N (não-base):\n{N}')

            x_nao_artificial = np.zeros(len(indices_nao_artificiais_novos))
            x_artificial = np.dot(B_inv, b)

            p = np.dot(c_padrao_2_fase[:, indices_artificiais_novos], B_inv)
            z = np.dot(p, b)

            solucao_geral = np.dot(B_inv, (np.dot(N, x_nao_artificial) + np.dot(B, x_artificial) - b))

            print(f"\nNovo valor da função objetivo: {np.dot(z[0][0], -1)}")
            print(f"Nova solução básica viável:")
            for i, var in enumerate(novas_variaveis_reordenadas):
                if i in indices_artificiais_novos:
                    print(f"{var[0]} = {x_artificial[indices_artificiais_novos.index(i)][0]}")
                else:
                    print(f"{var[0]} = {x_nao_artificial[indices_nao_artificiais_novos.index(i)]}")
            print(f"Nova solução geral primeira interação:\n{solucao_geral}")

            logging.debug(f"\nNovo valor da função objetivo: {np.dot(z[0][0], -1)}")
            logging.info(f"Nova solução básica viável:")
            for i, var in enumerate(novas_variaveis_reordenadas):
                if i in indices_artificiais_novos:
                    logging.debug(f"{var[0]} = {x_artificial[indices_artificiais_novos.index(i)][0]}")
                else:
                    logging.debug(f"{var[0]} = {x_nao_artificial[indices_nao_artificiais_novos.index(i)]}")

    else:
        print("Não foi possível calcular x_base devido a erro de inversão.")
        logging.error("Não foi possível calcular x_base devido a erro de inversão.")

else:
    print("O problema NÃO precisa de variáveis artificiais.")
    logging.info("O problema NÃO precisa de variáveis artificiais.")

# Verificando se a matriz B (base) é invertível
if tem_artificiais:
    print("\n===== SIMPLEX - FASE 2 =====")
    logging.info("Simplex fase 2")
    for i in novas_variaveis:
        if 'w' in str(i):
            novas_variaveis = np.delete(novas_variaveis, np.where(novas_variaveis == i)[0][0], axis=0)

    indices_novas_variaveis = [i for i in range(len(novas_variaveis))]

    A_padrao = A_reordenada[:, indices_novas_variaveis].astype(np.float64)
    c_padrao = c_padrao[:, indices_novas_variaveis].flatten().tolist()
    c_padrao = np.array(c_padrao).reshape(1, -1)

    print(f"\nMatriz A (sem variáveis artificiais):\n{A_padrao}")
    print(f"\nVetor c (sem variáveis artificiais):\n{c_padrao}")
    
    indices_base = list(ultimos_indices_artificiais)
    indices_nao_base = list([i for i in range(len(novas_variaveis)) if i not in indices_base])

else:
    print("\n===== SIMPLEX - PRIMAL =====")
    logging.info("Simplex")
    
print("\nResolvendo o problema:")
logging.info("Investendo a matriz B")
try:
    B = A_padrao[:, indices_base].astype(np.float64)
    print(f'\n - Matriz B (base):\n{B}')
    B_inv = np.linalg.inv(B)
except np.linalg.LinAlgError:
    print("Erro: a matriz B não é invertível.")
    logging.error("Matriz B não é invertível")
    B_inv = None

if B_inv is not None:
    N = A_padrao[:, indices_nao_base]
    print(f'\n - Matriz B (base):\n{B}')
    print(f'- Matriz N (não-base):\n{N}')

    # Cálculo da solução básica viável
    print("\n - Resolvendo o sistema para encontrar solução básica viável")
    logging.info("Resolvendo o sistema para encontrar solução básica viável")
    x_nao_base = np.zeros(len(indices_nao_base))
    x_base = np.dot(B_inv, b)

    p = np.dot(c_padrao[:, indices_base].reshape(1, -1), B_inv)
    z = np.dot(p, b) + sum((c_padrao[:, indices_nao_base] - np.dot(p, N[:, j])) * x_nao_base[j] for j in range(len(indices_nao_base)))

    solucao_geral = np.dot(B_inv, (np.dot(N, x_nao_base) + np.dot(B, x_base) - b))

    print(f'Valor da função objetivo: {np.dot(z[0][0],-1)}')
    print("Solução básica viável:")
    for i, var in enumerate(novas_variaveis):
        if i in indices_base:
            print(f"{var[0]} = {x_base[indices_base.index(i)][0]}")
        else:
            print(f"{var[0]} = {x_nao_base[indices_nao_base.index(i)]}")
    print(f"Resultado solução geral:\n{solucao_geral}")

    logging.debug(f'Valor da função objetivo: {np.dot(z[0][0],-1)}')
    logging.info("Solução básica viável:")
    for i, var in enumerate(novas_variaveis):
        if i in indices_base:
            logging.debug(f"{var[0]} = {x_base[indices_base.index(i)][0]}")
        else:
            logging.debug(f"{var[0]} = {x_nao_base[indices_nao_base.index(i)]}")

    # Tem como melhorar a solução?
    iteracao = 0
    max_iteracoes = 10
    while iteracao < max_iteracoes:
        iteracao += 1
        print(f"\n=== ITERAÇÃO {iteracao} ===")
        logging.info(f"Interação {iteracao}")
        print("\n - Tem como melhorar a solução básica?")

        s_j = [c_padrao[0][j] - np.dot(p, N[:, i]) for i, j in enumerate(indices_nao_base)]
        print("Custos reduzidos:")
        logging.info("Custos reduzido:")
        for i, j in enumerate(indices_nao_base):
            print(f"s{j + 1} = {s_j[i]}")
            logging.debug(f"s{j + 1} = {s_j[i]}")

        epsilon = 1e-8

        menor_c = np.min(s_j)
        print(f"Menor valor em c: {menor_c}")
        logging.debug(f"Menor valor em c: {menor_c}")
        if menor_c < -epsilon:
            print("Sim, é possível melhorar a solução.")
            logging.debug("Sim, é possível melhorar a solução.")
        else:
            print("Não, não é possível melhorar a solução.")
            logging.error("Não, não é possível melhorar a solução.")
            print("\n ===== SOLUÇÃO ÓTIMA DO PRIMAL ENCONTRADA ======")
            logging.info("Solução Otima do Primal:")
            print(f"\nSolução ótima encontrada na iteração {iteracao}:")
            print(f"Valor ótimo da função objetivo: {np.dot(z[0][0], -1)}")
            print("Solução ótima:")
            for i, var in enumerate(novas_variaveis):
                if i in indices_base:
                    print(f"{var[0]} = {x_base[indices_base.index(i)][0]}")
                else:
                    print(f"{var[0]} = {x_nao_base[indices_nao_base.index(i)]}")
            break

            logging.debug(f"Valor ótimo da função objetivo: {np.dot(z[0][0], -1)}")
            logging.info("Solução ótima:")
            for i, var in enumerate(novas_variaveis):
                if i in indices_base:
                    logging.debug(f"{var[0]} = {x_base[indices_base.index(i)][0]}")
                else:
                    logging.debug(f"{var[0]} = {x_nao_base[indices_nao_base.index(i)]}")
            break

        indice_entrada_nao_base = np.argmin(s_j)
        indice_entrada = indices_nao_base[indice_entrada_nao_base]

        if menor_c >= -epsilon:
            print("Não há mais variáveis que podem entrar na base.")
            logging.info("Não há mais variáveis que podem entrar na base.")
        var_entrada = novas_variaveis[indice_entrada]
        print(f"\nVariável que vai entrar na base: {var_entrada}")
        logging.debug(f"\nVariável que vai entrar na base: {var_entrada}")

        y = np.dot(B_inv, A_padrao[:, indice_entrada])

        razoes = []
        for i in range(len(indices_base)):
            x_base_val = x_base[i][0]
            if y[i] > 0:
                razoes.append(x_base_val / y[i])
            else:
                razoes.append(np.inf)

        min_razao = min(razoes)
        if min_razao == np.inf:
            print("Não há solução viável, pois não há limite superior.")
            logging.error("Não há solução viável, pois não há limite superior.")
            break
        indice_saida = np.argmin(razoes)
        var_saida = novas_variaveis[indices_base[indice_saida]]
        print(f"Variável que vai sair da base: {var_saida}")
        logging.debug(f"Variável que vai sair da base: {var_saida}")
        print(f"Valor máximo que a variável de entrada pode assumir: {min_razao}")
        logging.debug(f"Valor máximo que a variável de entrada pode assumir: {min_razao}")

        indices_base[indice_saida] = indice_entrada
        indices_nao_base = [j for j in range(len(novas_variaveis)) if j not in indices_base]

        B = A_padrao[:, indices_base].astype(np.float64)
        B_inv = np.linalg.inv(B)
        N = A_padrao[:, indices_nao_base]
        print(f'\n - Matriz B (base):\n{B}')
        print(f'- Matriz N (não-base):\n{N}')

        x_nao_base = np.zeros(len(indices_nao_base))
        x_base = np.dot(B_inv, b)

        p = np.dot(c_padrao[:, indices_base], B_inv)
        z = np.dot(p, b)

        solucao_geral = np.dot(B_inv, (np.dot(N, x_nao_base) + np.dot(B, x_base) - b))

        print(f"\nNovo valor da função objetivo: {np.dot(z[0][0], -1)}")
        print(f"Nova solução básica viável:")
        for i, var in enumerate(novas_variaveis):
            if i in indices_base:
                print(f"{var[0]} = {x_base[indices_base.index(i)][0]}")
            else:
                print(f"{var[0]} = {x_nao_base[indices_nao_base.index(i)]}")
        print(f"Nova solução geral primeira interação:\n{solucao_geral}")

        logging.debug(f"\nNovo valor da função objetivo: {np.dot(z[0][0], -1)}")
        logging.info(f"Nova solução básica viável:")
        for i, var in enumerate(novas_variaveis):
            if i in indices_base:
                logging.debug(f"{var[0]} = {x_base[indices_base.index(i)][0]}")
            else:
                logging.debug(f"{var[0]} = {x_nao_base[indices_nao_base.index(i)]}")

else:
    print("Não foi possível calcular x_base devido a erro de inversão.")
    logging.erro("Não foi possível calcular x_base devido a erro de inversão.")

print("\n====== SOLUÇÃO DO DUAL ======")
logging.info("Solução do Dual:")
print(f"\nSolução do dual:")
print(f"Valor ótimo da função objetivo: {np.dot(z[0][0], -1)}")
logging.debug(f"Valor ótimo da função objetivo: {np.dot(z[0][0], -1)}")
print("Solução (valor sombra):")
p = np.dot(c_padrao[:, indices_base], B_inv)
max_limiting_value = float('-inf')
max_limiting_index = -1

for i, value in enumerate(p.flatten(), start=1):
    if np.isclose(value, 0):
        print(f"y{i} = 0 (Componente em abundância)")
        logging.debug(f"y{i} = 0 (Componente em abundância)")
    else:
        adjusted_value = abs(value)
        print(f"y{i} = {adjusted_value} (Componente que limita a produção)")
        logging.debug(f"y{i} = {adjusted_value} (Componente que limita a produção)")
        if adjusted_value > max_limiting_value:
            max_limiting_value = adjusted_value
            max_limiting_index = i

if max_limiting_index != -1:
    print(f"O componente que mais limita a produção é y{max_limiting_index}")
    logging.info(f"O componente que mais limita a produção é y{max_limiting_index}")

print("\n====== ANALISE DE SENSIBILIDADE ======")
logging.info("Fazendo analise de sensibilidade.")
print("\nAnalise de sensibilidade:")

indices = indices_base + indices_nao_base
indices_variaveis = [i for i, _ in enumerate(variaveis)]
valores = [
    x_base[indices_base.index(i)][0] if i in indices_base else x_nao_base[indices_nao_base.index(i)]
    for i in sorted(indices)
]
valores_x = [valores[i] for i in indices_variaveis]

print("\nAnálise das restrições:")
for i, row in enumerate(A):
    result = np.dot(row, valores_x)
    y_value = p[0][i]
    status = "abundante" if np.isclose(y_value, 0) else "limitante"
    print(f"Restrição {i + 1} | {result:.4f} {sinais[i]} {b[i][0]} | y{i + 1} = {y_value:.4f} ({status})")
    logging.debug(f"Restrição {i + 1} | {result:.4f} {sinais[i]} {b[i][0]} | y{i + 1} = {y_value:.4f} ({status})")

print("\nAnalise dos coeficientes:")
for i, var in enumerate(novas_variaveis):
    if i < len(variaveis):
        valor_variavel = x_base[indices_base.index(i)][0] if i in indices_base else 0
        custo_reduzido = c_padrao[0][i] - np.dot(p, A_padrao[:, i])
        coeficiente = c_padrao[0][i]
        print(f"Variável {var[0]} = {valor_variavel:.4f} | Custo reduzido = {custo_reduzido[0]} | Coeficiente = {np.dot(coeficiente, -1)}")
        logging.debug(f"Variável {var[0]} = {valor_variavel:.4f} | Custo reduzido = {custo_reduzido[0]} | Coeficiente = {np.dot(coeficiente, -1)}")

print("\nVariações Restrições:")
valores_basicos = [x_base[i][0] for i in range(len(x_base))]

for i in range(len(b)):
    coluna = B_inv[:, i]

    diminuicoes = [valores_basicos[j] / coluna[j] for j in range(len(coluna)) if coluna[j] > 0]
    aumentos = [valores_basicos[j] / coluna[j] for j in range(len(coluna)) if coluna[j] < 0]

    max_aumento = -max(aumentos) if aumentos else np.inf
    max_diminuicao = min(diminuicoes) if diminuicoes else np.inf

    print(f"Recurso {i+1}:")
    print(f"  Pode aumentar: {max_aumento:.2f}")
    print(f"  Pode diminuir: {max_diminuicao:.2f}")