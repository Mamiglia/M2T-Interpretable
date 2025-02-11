import argparse
import os
import pickle
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from datasets.loader import build_data
from datasets.vocabulary import vocabulary

BASE_PATH = "/media/hdd/usr/tao/momask-codes/"

def load_model_config(args, device):
    # Load model configuration and weights
    if "kit" in args.dataset_name:
        from architectures.LSTM_kit import seq2seq
    elif args.dataset_name == "h3D":
        from architectures.LSTM_h3D import seq2seq

    loaded_model = seq2seq(
        args.vocab_size, args.hidden_size, args.embedding_dim, num_layers=1, device=device,
        attention=args.attention_type, beam_size=args.beam_size, hidden_dim=args.hidden_dim, K=args.K
    ).to(device)

    model_dict = torch.load(args.path, map_location=device)
    loaded_model.load_state_dict(model_dict["model"])
    return loaded_model

def load_data(args):
    project_path = BASE_PATH + "dataset/HumanML3D"
    aug_path = BASE_PATH + "dataset/HumanML3D"

    if "kit" in args.dataset_name:
        from datasets.kit_h3mld import dataset_class
        path_txt = os.path.join(project_path, "datasets/sentences_corrections.csv")
        path_motion = os.path.join(aug_path, "kit_with_splits_2023.npz")
    elif args.dataset_name == "h3D":
        from datasets.h3d_m2t_dataset_ import dataset_class
        path_txt = os.path.join(aug_path, "sentences_corrections_h3d.csv")
        path_motion = os.path.join(aug_path, "all_humanML3D.npz")

    train_data_loader, val_data_loader, test_data_loader = build_data(
        dataset_class=dataset_class, min_freq=args.min_freq, path=path_motion,
        train_batch_size=args.batch_size, test_batch_size=args.batch_size,
        return_lengths=True, path_txt=path_txt, return_trg_len=True,
        joint_angles=False, multiple_references=args.multiple_references
    )

    return train_data_loader, val_data_loader, test_data_loader

def get_vocab_size(args):
    train_data_loader, _, _ = load_data(args)
    return train_data_loader.dataset.lang.vocab_size_unk

def get_vocab(args):
    data = pickle.load(open(args.dataset_path, 'rb'))
    
    sentences = data['old_desc']
    sentences = [d for ds in sentences for d in ds] #flat all descriptions in order
    
    correct_tokens = False
    vocab = vocabulary(
        sentences=sentences, correct_tokens=correct_tokens
    )
    vocab.build_vocabulary(min_freq=args.min_freq)
    assert vocab.vocab_size == args.vocab_size, f"Vocab size mismatch: {vocab.vocab_size} != {args.vocab_size}"
    return vocab

def preprocess_motions(input_folder):
    motions = []
    for file in os.listdir(input_folder):
        if file.endswith(".npy"):
            motion = np.load(os.path.join(input_folder, file))
            motions.append(torch.tensor(motion, dtype=torch.float32))
    return pad_sequence(motions, batch_first=True)

@torch.no_grad()
def perform_inference(model, motions, device):
    model.eval()
    
    B, T, J, _ = motions.shape
    src = motions.view(B,T,J*3).permute(1, 0, 2).to(device) # T x B x (3J)
    src_lens = [len(motion) for motion in motions]
    trg = torch.zeros((src.size(0), src.size(1)), dtype=torch.long).to(device)
    output = model(src, trg, teacher_force_ratio=0, src_lens=src_lens)
    return output

def save_captions(captions, output_folder, vocab: vocabulary):
    # postprocess captions
    print('caption shape:', captions.shape)
    tokens_id = captions.argmax(dim=-1).T
        
    os.makedirs(output_folder, exist_ok=True)
    for i, caption in enumerate(tokens_id):
        caption = vocab.decode_numeric_sentence(caption.tolist(), remove_sos_eos=True, ignore_pad=True)
        print(caption)
        with open(os.path.join(output_folder, f"caption_{i}.txt"), "w") as f:
            f.write(caption)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing input motion npy files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the generated captions")
    parser.add_argument("--path", type=str, required=True, help="Path of model weights")
    parser.add_argument("--dataset_name", type=str, default="h3D", choices=["h3D", "kit"])
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--attention_type", type=str, default="relative_bahdanau")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--min_freq", type=int, default=3)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--K", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=3605)
    parser.add_argument("--multiple_references", type=bool, default=False)
    parser.add_argument("--dataset_path", type=str, default="/media/hdd/usr/tao/momask-codes/dataset/HumanML3D/all_humanML3D.npz")
    args = parser.parse_args()
        
        
    device = torch.device(args.device)
    # train, val, test = load_data(args)
    vocab = get_vocab(args)
    model = load_model_config(args, device)
    motions = preprocess_motions(args.input_folder)
    captions = perform_inference(model, motions, device)
    save_captions(captions, args.output_folder, vocab)
