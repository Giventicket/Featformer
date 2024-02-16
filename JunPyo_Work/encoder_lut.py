import torch
import torch.nn.functional as F
from tqdm import tqdm

embedding1_cache = None
grid_coordinates = None
def get_embedding1(tsp_instance, grid_size = 256):

    """
        tsp_instance: single tsp instance (N, 2)
        grid_size: whole grid size for embedding1
    """
    
    global embedding1_cache, grid_coordinates
    
    # precalculate embedding1_cache only once.
    if embedding1_cache is None:
        depth_max = 4 # depth max is fixed.
        maps_for_embedding1 = {}
        for depth in range(1, depth_max + 1):
            maps_for_embedding1[depth] = torch.zeros(grid_size, grid_size)
            
        # function in function due to simplicity
        def recur(x, y, depth, index, grid_size):
            cur_grid_size = grid_size // (2 ** depth)

            if depth > 0:
                maps_for_embedding1[depth][y:y+cur_grid_size, x:x+cur_grid_size] = index

            if depth < depth_max:
                next_grid_size = cur_grid_size // 2
                recur(x, y, depth + 1, 0, grid_size)
                recur(x + next_grid_size, y, depth + 1, 1, grid_size)
                recur(x, y + next_grid_size, depth + 1, 2, grid_size)
                recur(x + next_grid_size, y + next_grid_size, depth + 1, 3, grid_size)

        recur(0, 0, 0, 0, grid_size)
        
        one_hots = []
        for depth in range(1, depth_max + 1):
            one_hot = F.one_hot(maps_for_embedding1[depth].to(torch.long), num_classes=4)
            one_hots.append(one_hot)
        
        embedding1_cache = torch.cat(one_hots, dim=-1).to("cuda")

    # precalculate grid_coordinates only once.
    if grid_coordinates is None:
        offset = 0.5 / grid_size
        grid_x, grid_y = torch.meshgrid(torch.linspace(offset, 1 - offset, grid_size), torch.linspace(offset, 1 - offset, grid_size))
        grid_coordinates = torch.stack((grid_x, grid_y), dim=-1).view(-1, 2).to("cuda")
    
    closest_indices = torch.cdist(tsp_instance.to("cuda"), grid_coordinates).argmin(dim=-1)
    
    rows = closest_indices // grid_size
    cols = closest_indices % grid_size
    
    embedding1 = embedding1_cache[rows, cols]
    return embedding1

def get_count_matrix(dist_matrix, dist_max):
    heatmap = dist_matrix <= dist_max
    count_matrix = heatmap.sum(dim = -1)
    return count_matrix

def get_one_hot_encoding(count_nodes, num_categories):
    # 0 ~ num_categories-1 의 category에 할당됨.
    bins = torch.linspace(-1, count_nodes.max() + 1, num_categories + 1).to("cuda")
    category_indices = torch.bucketize(count_nodes, bins, right = False) - 1
    one_hot_categories = torch.nn.functional.one_hot(category_indices, num_categories)
    return one_hot_categories

def get_embedding2(dist_matrix):
    max_distance = dist_matrix.max()
    
    count_matrix1 = get_count_matrix(dist_matrix, max_distance / 16) # range1 구간의 개수 counting, [N]
    count_matrix2 = get_count_matrix(dist_matrix, max_distance / 8) # range2 구간의 개수 counting, [N]
    
    one_hot1 = get_one_hot_encoding(count_matrix1, 8)
    one_hot2 = get_one_hot_encoding(count_matrix2, 8)
    
    embedding2 = torch.concat([one_hot1, one_hot2], dim = -1)
    return embedding2

def get_angle_matrix(node_positions):
    delta_xs = node_positions[:, 0].unsqueeze(dim = 1) - node_positions[:, 0].unsqueeze(dim = -1)
    delta_ys = node_positions[:, 1].unsqueeze(dim = 1) - node_positions[:, 1].unsqueeze(dim = -1)
    angle_matrix = torch.atan2(delta_ys, delta_xs)
    return angle_matrix

def get_embedding3(dist_matrix, angle_matrix):
    num_sectors = 12
    sector_angle = 2 * torch.pi / num_sectors
    
    one_hots = []
    max_distance = dist_matrix.max()
    
    for i in range(num_sectors):
        sector_start = -torch.pi + i * sector_angle
        sector_end = -torch.pi + (i + 1) * sector_angle
        in_sector_mask = (angle_matrix > sector_start) & (angle_matrix <= sector_end)
        
        count_matrix = get_count_matrix(dist_matrix * in_sector_mask, max_distance / 8)
        one_hot = get_one_hot_encoding(count_matrix, 8)
        one_hots.append(one_hot)
        
    embedding3 = torch.concat(one_hots, dim = -1)
    return embedding3
    

def get_encoder_embedding(tsp_instance):
    dist_matrix = torch.cdist(tsp_instance.to("cuda"), tsp_instance.to("cuda"))
    angle_matrix = get_angle_matrix(tsp_instance.to("cuda"))
    
    embedding1 = get_embedding1(tsp_instance)
    embedding2 = get_embedding2(dist_matrix)
    embedding3 = get_embedding3(dist_matrix, angle_matrix)
    
    embedding = torch.concat([embedding1, embedding2, embedding3], dim = -1)
    return embedding.cpu()

if __name__ == "__main__":    
    num_samples, size, node_dim = 100000, 100, 2
    data = torch.rand(num_samples, size, node_dim)
    
    for batch_idx, tsp_instance in tqdm(enumerate(data)):
        embedding = get_encoder_embedding(tsp_instance)
        print(embedding.shape)