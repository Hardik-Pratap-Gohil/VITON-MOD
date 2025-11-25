"""
Training script for Frequency-Enhanced VITON-HD

This script implements training for the novel frequency-domain approach
to virtual try-on with improved detail preservation.

Usage:
    python train.py --name experiment_name --dataset_dir ./datasets/
"""

import argparse
import os
import time
from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from networks import SegGenerator, FrequencyEnhancedGMM, ALIASGenerator
from datasets import VITONDataset, VITONDataLoader
from losses import CombinedLoss, SegmentationLoss
from utils import load_checkpoint, gen_noise, save_images


def get_opt():
    parser = argparse.ArgumentParser()

    # Experiment name
    parser.add_argument("--name", type=str, required=True, help="Experiment name")

    # Dataset parameters
    parser.add_argument("--dataset_dir", type=str, default="./datasets/")
    parser.add_argument("--dataset_mode", type=str, default="train", help="train or test")
    parser.add_argument("--dataset_list", type=str, default="train_pairs.txt")

    # Model checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--load_pretrained", action="store_true", help="Load pretrained models")
    parser.add_argument("--seg_checkpoint", type=str, default="seg_final.pth")
    parser.add_argument("--gmm_checkpoint", type=str, default="gmm_final.pth")
    parser.add_argument("--alias_checkpoint", type=str, default="alias_final.pth")

    # Training parameters
    parser.add_argument("--train_mode", type=str, default="all",
                       choices=["seg", "gmm", "alias", "all"],
                       help="Which module to train: seg, gmm, alias, or all")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")

    # Loss weights
    parser.add_argument("--lambda_l1", type=float, default=1.0, help="Weight for L1 loss")
    parser.add_argument("--lambda_freq", type=float, default=0.5, help="Weight for frequency loss (NOVEL)")
    parser.add_argument("--lambda_perceptual", type=float, default=0.1, help="Weight for perceptual loss")
    parser.add_argument("--lambda_style", type=float, default=0.05, help="Weight for style loss")

    # Model architecture
    parser.add_argument("--load_height", type=int, default=1024)
    parser.add_argument("--load_width", type=int, default=768)
    parser.add_argument("--semantic_nc", type=int, default=13, help="Number of segmentation classes")
    parser.add_argument("--grid_size", type=int, default=5, help="TPS grid size")
    parser.add_argument("--ngf", type=int, default=64, help="Number of generator filters")
    parser.add_argument("--norm_G", type=str, default="spectralaliasinstance")
    parser.add_argument("--num_upsampling_layers", type=str, default="most",
                       choices=["normal", "more", "most"])
    parser.add_argument("--init_type", type=str, default="xavier",
                       choices=["normal", "xavier", "xavier_uniform", "kaiming", "orthogonal"])
    parser.add_argument("--init_variance", type=float, default=0.02)

    # Logging and saving
    parser.add_argument("--display_freq", type=int, default=100, help="Display frequency")
    parser.add_argument("--save_freq", type=int, default=1000, help="Save checkpoint frequency")
    parser.add_argument("--save_dir", type=str, default="./results/")

    opt = parser.parse_args()
    return opt


def train_segmentation(opt, seg_model, dataloader, device):
    """
    Train segmentation generator.
    Predicts cloth region segmentation from agnostic inputs.
    """
    print("\n" + "="*60)
    print("Training Segmentation Generator")
    print("="*60)

    seg_model.to(device).train()

    # Optimizer
    optimizer = torch.optim.Adam(seg_model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Loss
    criterion = SegmentationLoss()

    # TensorBoard
    writer = SummaryWriter(osp.join('runs', opt.name, 'seg'))

    total_steps = 0

    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()

        for i, data in enumerate(dataloader):
            total_steps += 1

            # Move data to device
            c = data['cloth']['unpaired'].to(device)
            cm = data['cloth_mask']['unpaired'].to(device)
            parse_agnostic = data['parse_agnostic'].to(device)
            pose = data['pose'].to(device)

            # Generate noise
            noise = gen_noise(c.size(0)).to(device)

            # Forward pass
            input_seg = torch.cat([cm, c * cm, parse_agnostic, pose, noise], dim=1)
            parse_pred = seg_model(input_seg)

            # Loss (we don't have ground truth parse for cloth, so this is self-supervised)
            # In practice, you'd need paired data with ground truth segmentations
            # For now, we'll use parse_agnostic as a proxy target
            target_seg = parse_agnostic.argmax(dim=1)
            loss = criterion(parse_pred, target_seg)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if total_steps % opt.display_freq == 0:
                print(f"Epoch [{epoch}/{opt.num_epochs}] Step [{i}/{len(dataloader)}] "
                      f"Loss: {loss.item():.4f}")
                writer.add_scalar('loss', loss.item(), total_steps)

            # Save checkpoint
            if total_steps % opt.save_freq == 0:
                save_path = osp.join(opt.checkpoint_dir, opt.name, f'seg_step_{total_steps}.pth')
                os.makedirs(osp.dirname(save_path), exist_ok=True)
                torch.save(seg_model.state_dict(), save_path)
                print(f"Saved checkpoint: {save_path}")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    # Save final model
    final_path = osp.join(opt.checkpoint_dir, opt.name, 'seg_final.pth')
    torch.save(seg_model.state_dict(), final_path)
    print(f"Training complete. Final model saved: {final_path}")
    writer.close()


def train_gmm(opt, gmm_model, dataloader, device):
    """
    Train Frequency-Enhanced GMM (NOVEL).
    Warps cloth with frequency-domain detail preservation.
    """
    print("\n" + "="*60)
    print("Training Frequency-Enhanced GMM (NOVEL)")
    print("="*60)

    gmm_model.to(device).train()

    # Optimizer
    optimizer = torch.optim.Adam(gmm_model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Loss: L1 for warp quality
    criterion_l1 = nn.L1Loss()

    # TensorBoard
    writer = SummaryWriter(osp.join('runs', opt.name, 'gmm'))

    total_steps = 0

    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()

        for i, data in enumerate(dataloader):
            total_steps += 1

            # Move data to device
            c = data['cloth']['unpaired'].to(device)
            cm = data['cloth_mask']['unpaired'].to(device)
            img_agnostic = data['img_agnostic'].to(device)
            parse_cloth = data['parse_cloth'].to(device)
            pose = data['pose'].to(device)

            # Downsample for GMM (256x192)
            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse_cloth, size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            cm_gmm = F.interpolate(cm, size=(256, 192), mode='nearest')

            # GMM input
            gmm_input = torch.cat([parse_cloth_gmm, pose_gmm, agnostic_gmm], dim=1)

            # Forward pass (NOVEL: with frequency-aware encoding)
            warped_cloth, warped_grid = gmm_model(gmm_input, c_gmm)

            # Loss: warped cloth should match original cloth in masked regions
            loss_l1 = criterion_l1(warped_cloth, c_gmm)

            # Additional loss: warped mask should match input mask
            warped_mask = F.grid_sample(cm_gmm, warped_grid, padding_mode='border', align_corners=True)
            loss_mask = criterion_l1(warped_mask, parse_cloth_gmm)

            total_loss = loss_l1 + 0.5 * loss_mask

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if total_steps % opt.display_freq == 0:
                print(f"Epoch [{epoch}/{opt.num_epochs}] Step [{i}/{len(dataloader)}] "
                      f"Total Loss: {total_loss.item():.4f} "
                      f"L1: {loss_l1.item():.4f} Mask: {loss_mask.item():.4f}")
                writer.add_scalar('loss/total', total_loss.item(), total_steps)
                writer.add_scalar('loss/l1', loss_l1.item(), total_steps)
                writer.add_scalar('loss/mask', loss_mask.item(), total_steps)

            # Save checkpoint
            if total_steps % opt.save_freq == 0:
                save_path = osp.join(opt.checkpoint_dir, opt.name, f'gmm_step_{total_steps}.pth')
                os.makedirs(osp.dirname(save_path), exist_ok=True)
                torch.save(gmm_model.state_dict(), save_path)
                print(f"Saved checkpoint: {save_path}")

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    # Save final model
    final_path = osp.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth')
    torch.save(gmm_model.state_dict(), final_path)
    print(f"Training complete. Final model saved: {final_path}")
    writer.close()


def train_alias(opt, seg_model, gmm_model, alias_model, dataloader, device):
    """
    Train ALIAS Generator with novel frequency-domain losses.
    """
    print("\n" + "="*60)
    print("Training ALIAS Generator with Frequency-Domain Losses")
    print("="*60)

    # Set models to appropriate modes
    seg_model.to(device).eval()  # Frozen
    gmm_model.to(device).eval()  # Frozen
    alias_model.to(device).train()  # Training

    # Optimizer
    optimizer = torch.optim.Adam(alias_model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Loss (NOVEL: includes frequency-domain loss)
    criterion = CombinedLoss(
        lambda_l1=opt.lambda_l1,
        lambda_freq=opt.lambda_freq,
        lambda_perceptual=opt.lambda_perceptual,
        lambda_style=opt.lambda_style,
        use_gpu=torch.cuda.is_available()
    ).to(device)

    # TensorBoard
    writer = SummaryWriter(osp.join('runs', opt.name, 'alias'))

    total_steps = 0

    for epoch in range(opt.num_epochs):
        epoch_start_time = time.time()

        for i, data in enumerate(dataloader):
            total_steps += 1

            # Move data to device
            c = data['cloth']['unpaired'].to(device)
            cm = data['cloth_mask']['unpaired'].to(device)
            img = data['img'].to(device)  # Target image
            img_agnostic = data['img_agnostic'].to(device)
            parse_agnostic = data['parse_agnostic'].to(device)
            pose = data['pose'].to(device)
            parse_cloth = data['parse_cloth'].to(device)

            # Part 1: Segmentation (frozen)
            with torch.no_grad():
                noise = gen_noise(c.size(0)).to(device)
                input_seg = torch.cat([cm, c * cm, parse_agnostic, pose, noise], dim=1)

                # Downsample for segmentation
                input_seg_down = F.interpolate(input_seg, size=(256, 192), mode='bilinear', align_corners=True)
                parse_pred_down = seg_model(input_seg_down)
                parse_pred = F.interpolate(parse_pred_down, size=(opt.load_height, opt.load_width),
                                          mode='bilinear', align_corners=True)

                # Group segmentation classes
                parse_old = F.softmax(parse_pred, dim=1)
                labels = {
                    0: ['background', [0]],
                    1: ['paste', [2, 4, 7, 8, 9, 10, 11]],
                    2: ['upper', [3]],
                    3: ['hair', [1]],
                    4: ['left_arm', [5]],
                    5: ['right_arm', [6]],
                    6: ['noise', [12]]
                }
                parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width).to(device)
                for j in range(len(labels)):
                    for label in labels[j][1]:
                        parse[:, j] += parse_old[:, label]

            # Part 2: Geometric Matching (frozen, NOVEL frequency-aware)
            with torch.no_grad():
                agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
                parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
                pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
                c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
                gmm_input = torch.cat([parse_cloth_gmm, pose_gmm, agnostic_gmm], dim=1)

                warped_c_gmm, warped_grid = gmm_model(gmm_input, c_gmm)

                # Upsample warped cloth
                warped_c = F.grid_sample(c, warped_grid, padding_mode='border', align_corners=True)
                warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border', align_corners=True)

            # Part 3: Try-on Synthesis (training)
            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat([parse, misalign_mask], dim=1)
            parse_div[:, 2:3] -= misalign_mask

            output = alias_model(torch.cat([img_agnostic, pose, warped_c], dim=1),
                                parse, parse_div, misalign_mask)

            # Loss (NOVEL: includes frequency-domain loss)
            loss_dict = criterion(output, img)
            total_loss = loss_dict['total']

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Logging
            if total_steps % opt.display_freq == 0:
                print(f"Epoch [{epoch}/{opt.num_epochs}] Step [{i}/{len(dataloader)}]")
                print(f"  Total: {total_loss.item():.4f} | L1: {loss_dict['l1'].item():.4f} | "
                      f"Freq: {loss_dict['frequency'].item():.4f} | "
                      f"Perc: {loss_dict['perceptual'].item():.4f} | "
                      f"Style: {loss_dict['style'].item():.4f}")

                writer.add_scalar('loss/total', total_loss.item(), total_steps)
                writer.add_scalar('loss/l1', loss_dict['l1'].item(), total_steps)
                writer.add_scalar('loss/frequency', loss_dict['frequency'].item(), total_steps)
                writer.add_scalar('loss/perceptual', loss_dict['perceptual'].item(), total_steps)
                writer.add_scalar('loss/style', loss_dict['style'].item(), total_steps)

            # Save checkpoint
            if total_steps % opt.save_freq == 0:
                save_path = osp.join(opt.checkpoint_dir, opt.name, f'alias_step_{total_steps}.pth')
                os.makedirs(osp.dirname(save_path), exist_ok=True)
                torch.save(alias_model.state_dict(), save_path)
                print(f"Saved checkpoint: {save_path}")

                # Save sample images
                save_dir = osp.join(opt.save_dir, opt.name, f'step_{total_steps}')
                os.makedirs(save_dir, exist_ok=True)
                save_images(output[:4], [f'sample_{k}.jpg' for k in range(min(4, output.size(0)))], save_dir)

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")

    # Save final model
    final_path = osp.join(opt.checkpoint_dir, opt.name, 'alias_final.pth')
    torch.save(alias_model.state_dict(), final_path)
    print(f"Training complete. Final model saved: {final_path}")
    writer.close()


def main():
    opt = get_opt()
    print(opt)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create directories
    os.makedirs(osp.join(opt.checkpoint_dir, opt.name), exist_ok=True)
    os.makedirs(osp.join(opt.save_dir, opt.name), exist_ok=True)

    # Dataset
    print("Loading dataset...")
    dataset = VITONDataset(opt)
    dataloader = VITONDataLoader(opt, dataset)
    print(f"Dataset loaded: {len(dataset)} samples")

    # Models
    print("\nInitializing models...")
    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = FrequencyEnhancedGMM(opt, inputA_nc=7, inputB_nc=3)  # NOVEL: Frequency-Enhanced
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    # Load pretrained if specified
    if opt.load_pretrained:
        print("Loading pretrained models...")
        if osp.exists(osp.join(opt.checkpoint_dir, opt.seg_checkpoint)):
            load_checkpoint(seg, osp.join(opt.checkpoint_dir, opt.seg_checkpoint))
            print(f"Loaded segmentation: {opt.seg_checkpoint}")
        if osp.exists(osp.join(opt.checkpoint_dir, opt.gmm_checkpoint)):
            load_checkpoint(gmm, osp.join(opt.checkpoint_dir, opt.gmm_checkpoint))
            print(f"Loaded GMM: {opt.gmm_checkpoint}")
        if osp.exists(osp.join(opt.checkpoint_dir, opt.alias_checkpoint)):
            load_checkpoint(alias, osp.join(opt.checkpoint_dir, opt.alias_checkpoint))
            print(f"Loaded ALIAS: {opt.alias_checkpoint}")

    # Train based on mode
    if opt.train_mode == "seg":
        train_segmentation(opt, seg, dataloader, device)
    elif opt.train_mode == "gmm":
        train_gmm(opt, gmm, dataloader, device)
    elif opt.train_mode == "alias":
        train_alias(opt, seg, gmm, alias, dataloader, device)
    elif opt.train_mode == "all":
        print("\n" + "="*60)
        print("Training all modules sequentially")
        print("="*60)
        train_segmentation(opt, seg, dataloader, device)
        train_gmm(opt, gmm, dataloader, device)
        train_alias(opt, seg, gmm, alias, dataloader, device)
    else:
        raise ValueError(f"Unknown train_mode: {opt.train_mode}")

    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
