import sys
sys.path.append('..')
import os
import torch # type: ignore
import torchvision # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import absl.flags
import absl.app
import utils.datasets as datasets
import utils.utils as utils

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_integer("batch_size_test", 3, "Number of samples for each image")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "dir path where datasets are stored")
absl.flags.mark_flag_as_required("path_model")
# added new flags
absl.flags.DEFINE_integer("target_class", -1, "Class index to filter memory set (-1 means no filter)")
absl.flags.DEFINE_boolean("random_memory", False, "Use a random memory set")
absl.flags.DEFINE_string("dir_save_suffix", "", "Optional suffix to append to the save directory")
# new flags - task 8 (strat 1: balanced memory, strat 2: KNN, strat 3: K means clustering)
absl.flags.DEFINE_boolean("balanced_memory", False, "Use a balanced memory set with equal samples per class")
absl.flags.DEFINE_boolean("knn_memory", False, "Use KNN to select memory samples similar to input")
absl.flags.DEFINE_boolean("kmeans_memory", False, "Use KMeans clustering to select memory samples")

FLAGS = absl.flags.FLAGS
# wrong indices for task 8 strategies
WRONG_INDICES = {4, 11, 14, 23, 28, 32, 43, 47, 55, 56, 59, 61, 62, 63, 67, 68, 70, 74, 77, 96}



def run(path:str,dataset_dir:str):
    """ Function to generate memory images for testing images using a given
    model. Memory images show the samples in the memory set that have an
    impact on the current prediction.

    Args:
        path (str): model path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))    
    # load model
    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint['modality']
    if modality not in ['memory','encoder_memory']:
        raise ValueError(f'Model\'s modality (model type) must be one of [\'memory\',\'encoder_memory\'], not {modality}.')
    dataset_name = checkpoint['dataset_name']
    model = utils.get_model( checkpoint['model_name'],checkpoint['num_classes'],model_type=modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    # load data
    train_examples = checkpoint['train_examples']
    if dataset_name == 'CIFAR10' or dataset_name == 'CINIC10':
        name_classes= ['airplane','automobile',	'bird',	'cat','deer','dog',	'frog'	,'horse','ship','truck']
    else:
        name_classes = range(checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    batch_size_test = FLAGS.batch_size_test
    _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=100,size_train=train_examples)
    memory_iter = iter(mem_loader)
    
    # function filters memory by class
    def get_filtered_memory(target_class=-1):
        all_mem_images = []
        all_mem_labels = []
        for mem_images, mem_labels in mem_loader:
            all_mem_images.append(mem_images)
            all_mem_labels.append(mem_labels)
        all_mem_images = torch.cat(all_mem_images)
        all_mem_labels = torch.cat(all_mem_labels)
        if target_class >= 0:
            mask = all_mem_labels == target_class
            filtered = all_mem_images[mask]
        else:
            filtered = all_mem_images
        idx = torch.randperm(len(filtered))[:100]
        return filtered[idx]

    # Strategy: Balanced Memory
    def get_balanced_memory(num_classes=10, samples_per_class=10):
        all_mem_images = []
        all_mem_labels = []
        for mem_images, mem_labels in mem_loader:
            all_mem_images.append(mem_images)
            all_mem_labels.append(mem_labels)
        all_mem_images = torch.cat(all_mem_images)
        all_mem_labels = torch.cat(all_mem_labels)
        selected = []
        for c in range(num_classes):
            mask = all_mem_labels == c
            class_images = all_mem_images[mask]
            idx = torch.randperm(len(class_images))[:samples_per_class]
            selected.append(class_images[idx])
        return torch.cat(selected)

    # Strategy: KNN
    def get_knn_memory(input_image, k=100):
        from sklearn.neighbors import NearestNeighbors
        all_mem_images = []
        for mem_images, _ in mem_loader:
            all_mem_images.append(mem_images)
        all_mem_images = torch.cat(all_mem_images)
        mem_flat = all_mem_images.view(len(all_mem_images), -1).numpy()
        input_flat = input_image.cpu().view(1, -1).numpy()
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(mem_flat)
        indices = nn.kneighbors(input_flat, return_distance=False)[0]
        return all_mem_images[indices]

    # Strategy: K means clustering
    def get_kmeans_memory(input_image, k=10):
        from sklearn.cluster import KMeans
        all_mem_images = []
        for mem_images, _ in mem_loader:
            all_mem_images.append(mem_images)
        all_mem_images = torch.cat(all_mem_images)
        mem_flat = all_mem_images.view(len(all_mem_images), -1).numpy()
        input_flat = input_image.cpu().view(1, -1).numpy()
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(mem_flat)
        distances = np.linalg.norm(kmeans.cluster_centers_ - input_flat, axis=1)
        nearest_cluster = np.argmin(distances)
        cluster_indices = np.where(cluster_labels == nearest_cluster)[0]
        selected = cluster_indices[:100]
        return all_mem_images[selected]

    #saving stuff
    dir_save = "../images/mem_images/"+dataset_name+"/"+modality+"/" + checkpoint['model_name'] + "/" + FLAGS.dir_save_suffix + "/"
    if not os.path.isdir(dir_save): 
        os.makedirs(dir_save)

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im


    for batch_idx, (images, labels) in enumerate(test_loader): # labels added
        print("Batch:{}/{}".format(batch_idx, len(test_loader)), end='\r')
        if FLAGS.balanced_memory:
            memory = get_balanced_memory().to(device)
        elif FLAGS.knn_memory or FLAGS.kmeans_memory:
            memory = torch.zeros(1).to(device)
        elif FLAGS.random_memory:
            memory = get_filtered_memory(-1)
        else:
            try:
                memory, _ = next(memory_iter)
            except StopIteration:
                memory_iter = iter(mem_loader)
                memory, _ = next(memory_iter)
                
        images = images.to(device)
        labels = labels.to(device) # added
        memory = memory.to(device)

        # compute output
        if not FLAGS.knn_memory and not FLAGS.kmeans_memory:
            outputs,rw = model(images,memory,return_weights=True)
            _, predictions = torch.max(outputs, 1)
            mem_val,memory_sorted_index = torch.sort(rw,descending=True)
        else:
            predictions = torch.zeros(len(images), dtype=torch.long)
            mem_val = torch.zeros(len(images), 1)
            memory_sorted_index = torch.zeros(len(images), 1, dtype=torch.long)

        # compute memory outputs
        #mem_val,memory_sorted_index = torch.sort(rw,descending=True)

        # task 8 
        batch_indices = set(range(batch_idx * batch_size_test, (batch_idx + 1) * batch_size_test))
        target_in_batch = batch_indices & WRONG_INDICES
        if not target_in_batch:
            continue
        wrong_indices = torch.tensor([i - batch_idx * batch_size_test for i in sorted(target_in_batch)])

        fig = plt.figure(figsize=(len(wrong_indices)*2, 4),dpi=300)
        columns = len(wrong_indices) #

        rows = 2
        for plot_pos, ind in enumerate(wrong_indices.tolist()): # loop thru ONLY wrong images
            if FLAGS.knn_memory:
                memory = get_knn_memory(images[ind]).to(device)
                new_output, rw_new = model(images[ind].unsqueeze(0), memory, return_weights=True)
                _, new_pred = torch.max(new_output, 1)
                mem_val_ind, memory_sorted_index_ind = torch.sort(rw_new, descending=True)
                m_ec = memory_sorted_index_ind[0][mem_val_ind[0]>0]
            elif FLAGS.target_class >= 0:
                memory = get_filtered_memory(labels[ind].item()).to(device)
                new_output, rw_new = model(images[ind].unsqueeze(0), memory, return_weights=True)
                _, new_pred = torch.max(new_output, 1)
                mem_val_ind, memory_sorted_index_ind = torch.sort(rw_new, descending=True)
                m_ec = memory_sorted_index_ind[0][mem_val_ind[0]>0]
            elif FLAGS.kmeans_memory:
                memory = get_kmeans_memory(images[ind]).to(device)
                new_output, rw_new = model(images[ind].unsqueeze(0), memory, return_weights=True)
                _, new_pred = torch.max(new_output, 1)
                mem_val_ind, memory_sorted_index_ind = torch.sort(rw_new, descending=True)
                m_ec = memory_sorted_index_ind[0][mem_val_ind[0]>0]
            else:
                m_ec = memory_sorted_index[ind][mem_val[ind]>0]

            # get reduced memory
            input_selected = images[ind].unsqueeze(0)
            reduced_mem = undo_normalization(memory[m_ec])
            npimg = torchvision.utils.make_grid(reduced_mem,nrow=4).cpu().numpy()

            # build and store image

            fig.add_subplot(rows, columns, plot_pos+1)
            plt.imshow((get_image(input_selected)* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
            #pred_to_show = new_pred[0] if FLAGS.target_class >= 0 else predictions[ind]
            #pred_to_show = new_pred[0] if (FLAGS.target_class >= 0 or FLAGS.knn_memory) else predictions[ind]
            pred_to_show = new_pred[0] if (FLAGS.target_class >= 0 or FLAGS.knn_memory or FLAGS.kmeans_memory) else predictions[ind]
            plt.title('Idx:{}\nTrue:{}\nPred:{}'.format(batch_idx*batch_size_test+ind, name_classes[labels[ind]], name_classes[pred_to_show]))
            plt.axis('off')
            ax2 = fig.add_subplot(rows, columns, columns+plot_pos+1)
            plt.imshow((np.transpose(npimg, (1,2,0))* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
            plt.title('Used Samples')
            plt.axis('off')
        fig.tight_layout()
        fig.savefig(dir_save+str(batch_idx*batch_size_test+ind)+".png")
        plt.close()
        print('Generated {}/{} images'.format(batch_idx,len(test_loader)),end='\r')


def main(argv):

    run(FLAGS.path_model,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)