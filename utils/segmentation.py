from typing import Tuple, List
import os
from datetime import datetime
import nltk
from scipy.ndimage import binary_erosion, binary_dilation
from sklearn.cluster import KMeans
from constants import *
import torch
import numpy as np
from PIL import Image
from scipy.ndimage import label
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

"""
Self-segmentation technique taken from Prompt Mixing: https://github.com/orpatashnik/local-prompt-mixing
"""


class Segmentor:
    def __init__(self, prompt: str, object_nouns: List[str], struct_num_segments: int = 5, style_num_segments: int = 3,
                 res: int = 32):
        self.prompt = prompt
        self.struct_prompt = 'cat, dog'
        self.struct_num_segments = struct_num_segments
        self.style_num_segments = style_num_segments
        self.resolution = res
        self.object_nouns = object_nouns
        tokenized_prompt = nltk.word_tokenize(prompt)
        tokenized_struct_prompt =nltk.word_tokenize(self.struct_prompt)
        forbidden_words = [word.upper() for word in ["photo", "image", "picture"]]
        self.nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt))
                      if pos[:2] == 'NN' and word.upper() not in forbidden_words]
        self.struct_nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_struct_prompt))
                             if pos[:2] == 'NN' and word.upper() not in forbidden_words]
        a=1

    def update_attention(self, attn, is_cross):
        res = int(attn.shape[2] ** 0.5)
        if is_cross:
            if res == 16:
                self.cross_attention_32 = attn
            elif res == 32:
                self.cross_attention_64 = attn
        else:
            if res == 32:
                self.self_attention_32 = attn
            elif res == 64:
                self.self_attention_64 = attn

    def __call__(self, *args, **kwargs):
        clusters = self.cluster()
        cluster2noun = self.cluster2noun(clusters)
        return cluster2noun


    def visualize_cluster_nouns(self, clusters, noun_assignments, title,step):
        """
        Visualize clusters with annotated nouns.

        Args:
        clusters (np.array): The cluster array where each element is a cluster id.
        noun_assignments (dict): A dictionary where keys are cluster ids and values are the most relevant nouns.
        title (str): The title for the plot.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(clusters, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')

        # Annotate each cluster with the corresponding noun
        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            # Find the centroid of the cluster to place the text
            indices = np.where(clusters == cluster_id)
            centroid_x = np.mean(indices[1])
            centroid_y = np.mean(indices[0])
            noun = noun_assignments.get(cluster_id, "BG")
            plt.text(centroid_x, centroid_y, f'{noun[1]}', color='red', ha='center', va='center')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if step % MOD_STEP == 0:
            plt.savefig(f"mask_fig/{title.replace(' ', '_').lower()}_{timestamp}_step_{step}.png")
            plt.close()
        # plt.show()

    def cluster(self, res: int = 32):
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64

        style1_attn = self_attn[STYLE1_INDEX].mean(dim=0).cpu().numpy()
        style1_kmeans = KMeans(n_clusters=self.style_num_segments, n_init=10).fit(style1_attn)
        style1_clusters = style1_kmeans.labels_.reshape(res, res)

        style2_attn = self_attn[STYLE2_INDEX].mean(dim=0).cpu().numpy()
        style2_kmeans = KMeans(n_clusters=self.style_num_segments, n_init=10).fit(style2_attn)
        style2_clusters = style2_kmeans.labels_.reshape(res, res)

        struct_attn = self_attn[STRUCT_INDEX].mean(dim=0).cpu().numpy()
        struct_kmeans = KMeans(n_clusters=self.struct_num_segments, n_init=10).fit(struct_attn)
        struct_clusters = struct_kmeans.labels_.reshape(res, res)

        return style1_clusters, style2_clusters, struct_clusters

    def visualize_clusters(self, clusters, title,step=1):
        plt.figure(figsize=(8, 8))
        plt.imshow(clusters, cmap='viridis')
        plt.colorbar()
        plt.title(title)
        plt.axis('off')
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"mask_fig/{title.replace(' ', '_').lower()}_{timestamp}_step_{step}.png")
        plt.close()
        # plt.show()

    def visualize_masks(self, mask, title, cmap='gray', step=1):
        file_title = f"{title.replace(' ', '_').lower()}"
        with open(f'masks/{file_title}.npy', 'wb') as f:
            np.save(f,mask)
        f.close()
        if step % MOD_STEP == 0:
            plt.figure()
            plt.imshow(mask, cmap=cmap)
            plt.colorbar()
            plt.title(title)
            plt.axis('off')
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            plt.savefig(f"mask_fig/{title.replace(' ', '_').lower()}_{timestamp}_step_{step}.png")
            plt.close()
        # plt.show()

    def cluster2noun(self, clusters, cross_attn, attn_index, num_segments, struct_flag=False, is_cross=True):
        result = {}
        res = int(cross_attn.shape[2] ** 0.5)
        if struct_flag:
            nouns_indices = [index for (index, word) in self.struct_nouns]
        else:
            nouns_indices = [index for (index, word) in self.nouns]
        cross_attn = cross_attn[attn_index].mean(dim=0).reshape(res, res, -1)
        nouns_maps = cross_attn.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]
        if is_cross:
            normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=0).repeat(2, axis=1)
        else:
            normalized_nouns_maps = np.zeros_like(nouns_maps)#.repeat(2, axis=0).repeat(2, axis=1)
        for i in range(nouns_maps.shape[-1]):
            if is_cross:
                curr_noun_map = nouns_maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)
            else:
                curr_noun_map = nouns_maps[:, :, i] #.repeat(2, axis=0).repeat(2, axis=1)
            normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()

        max_score = 0
        all_scores = []
        for c in range(num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            all_scores.append(max(scores))
            max_score = max(max(scores), max_score)

        all_scores.remove(max_score)
        mean_score = sum(all_scores) / len(all_scores)

        for c in range(num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            if struct_flag:
                result[c] = self.struct_nouns[np.argmax(np.array(scores))] if max(scores) > 1.4 * mean_score else "BG"
            else:
                result[c] = self.nouns[np.argmax(np.array(scores))] if max(scores) > 1.4 * mean_score else "BG"

        return result

    def create_mask(self, clusters, cross_attention, attn_index, num_segments, struct_flag=False, is_cross=True):
        cluster2noun = self.cluster2noun(clusters, cross_attention, attn_index, num_segments, struct_flag=struct_flag,is_cross=is_cross)
        mask = clusters.copy()
        obj_segments = [c for c in cluster2noun if cluster2noun[c][1] in self.object_nouns]
        for c in range(num_segments):
            mask[clusters == c] = 1 if c in obj_segments else 0
        return torch.from_numpy(mask).to("cuda")

    def split_struct_by_noun(self, struct_clusters, struct_nouns):
        np.random.seed(1)
        # Extract nouns and cluster IDs into a dictionary
        noun_assignments = {cluster_id: noun for cluster_id, noun in struct_nouns}

        # Map each unique noun to an index
        unique_nouns = {noun: idx for idx, noun in enumerate(set(noun for _, noun in struct_nouns))}

        # Initialize a new array for annotated clusters
        annotated_clusters = np.zeros_like(struct_clusters, dtype=int)  # use int type for indices

        # Assign indices of nouns to their respective clusters
        for cluster_id, noun in noun_assignments.items():
            if noun in unique_nouns:
                noun_index = unique_nouns[noun]
                annotated_clusters[struct_clusters == cluster_id] = noun_index

        # Call the visualization method, passing also the index mapping
        self.visualize_cluster_nouns_split(annotated_clusters, unique_nouns, 'struct_by_noun', step='split')
        return
        # return object1_mask, object2_mask

    def visualize_cluster_nouns_split(self, clusters, unique_nouns, title, step='split'):
        """
        Visualize clusters with indices representing nouns, converted back to noun strings for display.
        Args:
            clusters (np.array): The cluster array where each element is an index representing a noun.
            unique_nouns (dict): A dictionary mapping nouns to indices.
            title (str): The title for the plot.
            step (str, optional): The step or phase of visualization, used to categorize the plot. Defaults to 'split'.
        """
        plt.figure(figsize=(10, 10))
        # Reverse the unique_nouns dictionary to map indices back to nouns for plotting
        index_to_noun = {idx: noun for noun, idx in unique_nouns.items()}

        # Iterate over unique indices in the clusters
        unique_indices = np.unique(clusters)
        for idx in unique_indices:
            if idx in index_to_noun:  # Ensure the index maps back to a noun
                indices = np.where(clusters == idx)
                centroid = (np.mean(indices[1]), np.mean(indices[0]))
                # Use the noun from the dictionary
                plt.text(centroid[0], centroid[1], index_to_noun[idx], color='white', ha='center', va='center',
                         fontsize=12)

        # Use a simple black and white color map and disable the color bar since it's not informative here
        plt.imshow(np.vectorize(lambda x: 0 if x == 0 else 1)(clusters), cmap='gray', interpolation='nearest')
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        plt.title(title)
        plt.savefig(f"mask_fig/{title.replace(' ', '_').lower()}_{timestamp}_step_{step}.png")
        plt.close()

    def split_structure_mask_objects_and_background(self, res=None):
        np.random.seed(1)
        self_attn = self.self_attention_32 if res == 32 else self.self_attention_64

        struct_attn = self_attn[STRUCT_INDEX].mean(dim=0).cpu().numpy()
        if struct_attn.ndim == 1:  # might need to remove
            # struct_attn = struct_attn.reshape(-1, 1)
            raise Exception
        struct_kmeans = KMeans(n_clusters=self.struct_num_segments, n_init=10).fit(struct_attn)
        struct_clusters = struct_kmeans.labels_.reshape(res, res)
        # Ensure background cluster values are always zero
        #max_cluster_label = np.max(struct_clusters) # will not function as intended
        #for i in range(res):
        #    for j in range(res):
        #        if struct_clusters[i][j] == max_cluster_label:
        #            struct_clusters[i][j] = 0

        # Save the mask array as an image with timestamped filename
        self.save_mask_as_image(struct_clusters, 'struct_clusters')
        self.split_struct_by_noun(struct_clusters, self.struct_nouns)
        return struct_clusters

    def save_mask_as_image(self, mask, base_filename):
        # Create a black-and-white image from the mask
        mask_image = Image.fromarray((mask * 255).astype(np.uint8),
                                     mode='L')  # Scale mask to 0-255 for visibility, mode='L' for grayscale
        # Invert the mask (black becomes white, white becomes black) to match the convention of white representing the object
        mask_image = Image.eval(mask_image, lambda x: 255 if x == 0 else 0)
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{base_filename}_{timestamp}.png"
        # Ensure the directory exists
        os.makedirs('splitting the masks', exist_ok=True)
        # Save the image
        mask_image.save(os.path.join('splitting the masks', filename))

    def split_structure_mask_objects(self, clustered_mask_objects):
        np.random.seed(1)
        if clustered_mask_objects.ndim != 2:
            raise ValueError("Input mask must be a 2D array.")

        unique, counts = np.unique(clustered_mask_objects, return_counts=True)
        background_label = unique[np.argmax(counts)]
        object_indices = np.argwhere(clustered_mask_objects != background_label)

        if len(object_indices) == 0:
            raise ValueError("No object pixels found for clustering.")

        # Reshape for KMeans
        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(object_indices)
        object_labels = kmeans.labels_

        mask1 = np.zeros_like(clustered_mask_objects, dtype=np.uint8)
        mask2 = np.zeros_like(clustered_mask_objects, dtype=np.uint8)

        for idx, label in enumerate(object_labels):
            if label == 0:
                mask1[object_indices[idx][0], object_indices[idx][1]] = 1
            else:
                mask2[object_indices[idx][0], object_indices[idx][1]] = 1

        # Convert masks to tensors (optional)
        mask1_tensor = torch.from_numpy(mask1)
        mask2_tensor = torch.from_numpy(mask2)

        # Save masks as images with timestamp
        self.save_mask_as_image(mask1, 'mask1')
        self.save_mask_as_image(mask2, 'mask2')

        return mask1_tensor, mask2_tensor

    def split_connected_components_and_save(self, mask, name, step, model_self=None, res=None, object_value=1,
                                            save_path='mask_fig/', connectivity=1, use_morphology=True):
        mask_np = mask.cpu().numpy()
        if mask_np.sum() == 0: # mask is completely empty
            if res==32:
                return model_self.object1_mask_32, model_self.object2_mask_32
            else: # res==64
                return model_self.object1_mask_64, model_self.object2_mask_64

        if use_morphology:
            # Initial settings for morphological operations
            erosion_size = 3  # Initial erosion size
            dilation_size = 2  # Initial dilation size

            while True:
                # Apply morphological erosion and dilation
                eroded_mask = binary_erosion(mask_np == object_value, structure=np.ones((erosion_size, erosion_size)))
                processed_mask = binary_dilation(eroded_mask, structure=np.ones((dilation_size, dilation_size)))

                # Find connected components
                labeled_mask, num_features = label(processed_mask,
                                                   structure=np.ones((3, 3)) if connectivity == 8 else None)

                if num_features >= 2:
                    break  # Exit the loop if successful
                else:
                    # Increase erosion and decrease dilation size to attempt further separation
                    erosion_size += 1
                    dilation_size = max(1, dilation_size - 1)
                    if erosion_size > 6:  # Avoid excessively large erosion which might remove all features
                        #print("Unable to separate components with morphological operations.")
                        if res == 32:
                            return model_self.object1_mask_32, model_self.object2_mask_32
                        else:  # res==64
                            return model_self.object1_mask_64, model_self.object2_mask_64
        else:
            processed_mask = mask_np == object_value
            labeled_mask, num_features = label(processed_mask, structure=np.ones((3, 3)) if connectivity == 8 else None)
            if num_features < 2:
                print("Less than two separate objects found.")
                if res == 32:
                    return model_self.object1_mask_32, model_self.object2_mask_32
                else:  # res==64
                    return model_self.object1_mask_64, model_self.object2_mask_64
                    # Proceed with saving the separated components
        fig, axes = plt.subplots(1, min(num_features, 2), figsize=(12, 6))
        masks = []
        for i in range(1, min(num_features, 2) + 1):
            mask_i = torch.from_numpy((labeled_mask == i).astype(int))
            masks.append(mask_i)
            ax = axes[i - 1] if num_features > 1 else axes
            ax.imshow(mask_i.cpu().numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f'Object {i} Mask')
            if i == 2 and step % MOD_STEP == 0:
                plt.savefig(f"{save_path}{name}_{step}_object{i}.png")
        plt.close(fig)
        return tuple(masks)
        
    def get_object_masks(self, is_cross=True, step=1, use_cluster = True) -> Tuple[torch.Tensor]:
        mask_style1_32, mask_style2_32, mask_struct_32, mask_style1_64, mask_style2_64, mask_struct_64 = tuple(torch.empty(0) for _ in range(6))
        flag_32 = hasattr(self, 'self_attention_32') or hasattr(self, 'cross_attention_32')
        flag_64 = hasattr(self, 'self_attention_64') or hasattr(self, 'cross_attention_64')
        if is_cross:
            title_addition = 'cross_attention'
            if flag_32:
                attn_32 = self.cross_attention_32
            if flag_64:
                attn_64 = self.cross_attention_64
        else:
            title_addition = 'self_attention'
            if flag_32:
                attn_32 = self.self_attention_32
            if flag_64:
                attn_64 = self.self_attention_64
        if flag_32:
            # Get cluster to noun mappings for visualization
            clusters_style1_32, clusters_style2_32, clusters_struct_32 = self.cluster(res=32)
            if use_cluster:
                cluster2noun_32_style1 = self.cluster2noun(clusters_style1_32, attn_32, STYLE1_INDEX, num_segments=self.style_num_segments, is_cross=is_cross)
                cluster2noun_32_style2 = self.cluster2noun(clusters_style2_32, attn_32, STYLE2_INDEX, num_segments=self.style_num_segments, is_cross=is_cross)
                cluster2noun_32_struct = self.cluster2noun(clusters_struct_32, attn_32, STRUCT_INDEX, num_segments=self.struct_num_segments,struct_flag=True,is_cross=is_cross)
                # Visualize clusters with nouns
                self.visualize_cluster_nouns(clusters_style1_32, cluster2noun_32_style1,
                                             f'{title_addition} Style1 Clusters Resolution 32 with Nouns step {step}',step=step)
                self.visualize_cluster_nouns(clusters_style2_32, cluster2noun_32_style2,
                                             f'{title_addition} Style2 Clusters Resolution 32 with Nouns step {step}', step=step)
                self.visualize_cluster_nouns(clusters_struct_32, cluster2noun_32_struct,
                                             f'{title_addition} Structural Clusters Resolution 32 with Nouns step {step}',step=step)

            mask_style1_32 = self.create_mask(clusters_style1_32, attn_32, STYLE1_INDEX, self.style_num_segments, is_cross=is_cross)
            mask_style2_32 = self.create_mask(clusters_style2_32, attn_32, STYLE2_INDEX, self.style_num_segments, is_cross=is_cross)
            mask_struct_32 = self.create_mask(clusters_struct_32, attn_32, STRUCT_INDEX, self.struct_num_segments, struct_flag=True, is_cross=is_cross)

            # Visualizing and saving the masks
        if flag_64:
            # Get cluster to noun mappings for visualization
            clusters_style1_64, clusters_style2_64, clusters_struct_64 = self.cluster(res=64)
            if use_cluster:
                cluster2noun_64_style1 = self.cluster2noun(clusters_style1_64, attn_64, STYLE1_INDEX, num_segments=self.style_num_segments,is_cross=is_cross)
                cluster2noun_64_style2 = self.cluster2noun(clusters_style2_64, attn_64, STYLE2_INDEX, num_segments=self.style_num_segments, is_cross=is_cross)
                cluster2noun_64_struct = self.cluster2noun(clusters_struct_64, attn_64, STRUCT_INDEX, num_segments=self.struct_num_segments,struct_flag=True, is_cross=is_cross)
                # Visualize clusters with nouns
                self.visualize_cluster_nouns(clusters_style1_64, cluster2noun_64_style1,
                                             f'{title_addition} Style1 Clusters Resolution 64 with Nouns',step=step)
                self.visualize_cluster_nouns(clusters_style2_64, cluster2noun_64_style2,
                                             f'{title_addition} Style2 Clusters Resolution 64 with Nouns', step=step)
                self.visualize_cluster_nouns(clusters_struct_64, cluster2noun_64_struct,
                                             f'{title_addition} Structural Clusters Resolution 64 with Nouns', step=step)

            mask_style1_64 = self.create_mask(clusters_style1_64, attn_64, STYLE1_INDEX, self.style_num_segments, is_cross=is_cross)
            mask_style2_64 = self.create_mask(clusters_style2_64, attn_64, STYLE1_INDEX, self.style_num_segments, is_cross=is_cross)
            mask_struct_64 = self.create_mask(clusters_struct_64, attn_64, STRUCT_INDEX, self.struct_num_segments, is_cross=is_cross)

            # Visualizing and saving the masks
            # self.visualize_masks(mask_style1_64.cpu().numpy(), f'{title_addition} Style1 Mask Resolution 64', step=step)
            # self.visualize_masks(mask_style2_64.cpu().numpy(), f'{title_addition} Style2 Mask Resolution 64',step=step)
            # self.visualize_masks(mask_struct_64.cpu().numpy(), f'{title_addition} Structural Mask Resolution 64',step=step)
        # Add visualization for clusters
        # self.visualize_clusters(clusters_style1_32, 'Style1 Clusters Resolution 32')
        # self.visualize_clusters(clusters_style2_32, 'Style2 Clusters Resolution 32')
        # self.visualize_clusters(clusters_struct_32, 'Structural Clusters Resolution 32')
        # self.visualize_clusters(clusters_style1_64, 'Style1 Clusters Resolution 64')
        # self.visualize_clusters(clusters_style2_64, 'Style2 Clusters Resolution 64')
        # self.visualize_clusters(clusters_struct_64, 'Structural Clusters Resolution 64')
        return mask_style1_32, mask_style2_32, mask_struct_32, mask_style1_64, mask_style2_64, mask_struct_64
