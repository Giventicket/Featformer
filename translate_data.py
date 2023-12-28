import pickle
import os
from tqdm import tqdm

def pickle_to_text(input_file_path, output_file_path):
    # Pickle 파일 읽기
    with open(input_file_path, "rb") as f:
        ls = pickle.load(f)

    # 텍스트로 변환하여 저장
    with open(output_file_path, "w") as f:
        for ele in tqdm(ls, desc=f"Converting {os.path.basename(input_file_path)}"):
            for x, y in ele:
                f.write(f"{x} {y} ")
            f.write("\n")

    print(f"Data has been dumped to {output_file_path}")

# 입력 및 출력 디렉토리 경로
input_directory = "/home/jpseo99/GeoFormer/trained_model_and_data/raw_data/"
output_directory = "/home/jpseo99/GeoFormer/trained_model_and_data/new_data/"

# 파일 목록
file_list = [
    "100_cluster.pkl", "100_expanded.pkl", "100_explosion.pkl", "100_grid.pkl", "100_implosion.pkl",
    "50_cluster.pkl", "50_expanded.pkl", "50_explosion.pkl", "50_grid.pkl", "50_implosion.pkl"
]

# 각 파일에 대해 변환 수행
for file_name in file_list:
    input_path = os.path.join(input_directory, file_name)
    output_path = os.path.join(output_directory, file_name.replace(".pkl", ".txt"))
    pickle_to_text(input_path, output_path)
