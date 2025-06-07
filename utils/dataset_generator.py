import os
import time
import torch
import requests
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset

# Função para baixar dados de cartas por raça, com verificação de imagem
def baixar_cartas_por_raca(raca, pasta_imagens="imagens"):
    url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?race={raca}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Erro ao acessar a API para {raca}: {response.status_code}")
        return []

    dados = response.json()

    if 'data' not in dados:
        print(f"Nenhuma carta encontrada para raça {raca}.")
        return []

    os.makedirs(f'{pasta_imagens}\\{raca}', exist_ok=True)
    dados_coletados = []

    for carta in tqdm(dados['data'], desc=f"Baixando {raca}"):
        if 'card_images' in carta:
            imagem_url = carta['card_images'][0]['image_url_cropped']
            nome_arquivo = f"{carta['id']}.jpg"
            caminho_imagem = os.path.join(f'{pasta_imagens}\\{raca}', nome_arquivo)

            sucesso = False
            for tentativa in range(2):  # Tenta até duas vezes
                try:
                    img_data = requests.get(imagem_url, timeout=10).content
                    with open(caminho_imagem, 'wb') as handler:
                        handler.write(img_data)

                    # Verifica se a imagem pode ser aberta
                    with Image.open(caminho_imagem) as img:
                        img.verify()
                    sucesso = True
                    break

                except Exception as e:
                    print(f"Tentativa {tentativa + 1} falhou para carta {carta['name']} ({carta['id']}): {e}")
                    if os.path.exists(caminho_imagem):
                        os.remove(caminho_imagem)

            if not sucesso:
                print(f"⚠️ Carta descartada após falhas: {carta['name']} ({carta['id']})")
                continue

            dados_coletados.append({
                'id': carta['id'],
                'nome': carta['name'],
                'raca': carta['race'],
                'caminho_imagem': caminho_imagem
            })

    return dados_coletados


# Dicionário para converter raças em índices numéricos
def mapear_racas(dados):
    racas_unicas = sorted(set(c['raca'] for c in dados))
    mapa = {raca: idx for idx, raca in enumerate(racas_unicas)}
    return mapa


# Dataset customizado para PyTorch
class YugiohDataset(Dataset):
    def __init__(self, dados, mapa_racas, transform=None):
        self.dados = dados
        self.transform = transform
        self.mapa_racas = mapa_racas

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        item = self.dados[idx]
        imagem = Image.open(item['caminho_imagem']).convert('RGB')
        if self.transform:
            imagem = self.transform(imagem)
        label = self.mapa_racas[item['raca']]
        return imagem, label


# Outro dataset baseado em DataFrame
class CartaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['caminho_imagem']

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['raca_codificada']

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    racas = [
        "Aqua", "Beast", "Beast-Warrior", "Dinosaur", "Dragon",
        "Fairy", "Fiend", "Fish", "Insect", "Machine", "Plant",
        "Psychic", "Pyro", "Reptile", "Rock", "Sea Serpent", "Spellcaster",
        "Thunder", "Warrior", "Winged Beast", "Wyrm", "Zombie",
        "Divine-Beast", "Illusion"
    ]

    todos_os_dados = []

    for raca in racas:
        dados = baixar_cartas_por_raca(raca)
        time.sleep(5)
        todos_os_dados.extend(dados)

    mapa_racas = mapear_racas(todos_os_dados)

    transformacoes = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = YugiohDataset(todos_os_dados, mapa_racas, transform=transformacoes)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    # Salvar dados em CSV
    pd.DataFrame(dataloader.dataset.dados).to_csv("db\\dados_cartas_yugioh.csv", index=False)
