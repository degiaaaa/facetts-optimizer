import os
from sacred import Experiment

from model.utils import fix_len_compatibility

ex = Experiment("face-tts")


@ex.config
def config():
    # Add this line to set a default value for perceptual_loss
    seed = int(os.getenv("seed", 37))
    perceptual_loss = int(os.getenv("perceptual_loss", 1))  # True: 1 / False: 0
    #local_checkpoint_dir = os.getenv("local_checkpoint_dir", "./checkpoints")

    # Dataset Configs
    # dataset = os.getenv("dataset", "lrs3")
    # lrs3_train = os.getenv("lrs3_train", "datalist/lrs3_train_long.list")
    # lrs3_val = os.getenv("lrs3_val", "datalist/lrs3_val_long.list")
    # lrs3_test = os.getenv("lrs3_test", "datalist/lrs3_test_long.list")
    # lrs3_path = os.getenv("lrs3_path", "data/lrs3")
    # cmudict_path = os.getenv("cmudict_path", "utils/cmu_dictionary")

    dataset = os.getenv("dataset", "lrs3")
    lrs3_train = os.getenv("lrs3_train", "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/datalist/lrs2_train_long.list")
    lrs3_val = os.getenv("lrs3_val", "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/datalist/lrs2_val_long.list")
    lrs3_test = os.getenv("lrs3_test", "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted/datalist/lrs2_test_long.list")
    lrs3_path = os.getenv("lrs3_path", "/mnt/qb/work2/butz1/bst080/data/mvlrs_v1/lrs2_splitted")
    cmudict_path = os.getenv("cmudict_path", "utils/cmu_dictionary")


    # Data Configs
    image_size = int(os.getenv("image_size", 224))
    max_frames = int(os.getenv("max_frames", 30))
    image_augment = int(os.getenv("image_augment", 0))

    ## hifigan-16k setting
    n_fft = int(os.getenv("n_fft", 1024))
    sample_rate = int(os.getenv("sample_rate", 16000))
    hop_len = int(os.getenv("hop_len", 160))
    win_len = int(os.getenv("win_len", 1024))
    f_min = float(os.getenv("f_min", 0.0))
    f_max = float(os.getenv("f_max", 8000))
    n_mels = int(os.getenv("n_mels", 128))
    # Network Configs

    ## Encoder parameters
    n_feats = n_mels
    spk_emb_dim = int(os.getenv("spk_emb_dim", 64))  # For multispeaker Grad-TTS
    vid_emb_dim = int(os.getenv("vid_emb_dim", 512))  # For Face-TTS
    n_enc_channels = int(os.getenv("n_enc_channels", 192))
    filter_channels = int(os.getenv("filter_channels", 768))
    filter_channels_dp = int(os.getenv("filter_channels_dp", 256))
    n_enc_layers = int(os.getenv("n_enc_layers", 6))
    enc_kernel = int(os.getenv("enc_kernel", 3))
    enc_dropout = float(os.getenv("enc_dropout", 0.0))
    n_heads = int(os.getenv("n_heads", 2))
    window_size = int(os.getenv("window_size", 4))

    ## Decoder parameters
    dec_dim = int(os.getenv("dec_dim", 64))
    beta_min = float(os.getenv("beta_min", 0.05))
    beta_max = float(os.getenv("beta_max", 20.0))
    pe_scale = float(os.getenv("pe_scale", 1000.0))

    ## Syncnet parameters
    syncnet_stride = int(os.getenv("syncnet_stride", 1))
    syncnet_ckpt = os.getenv("syncnet_ckpt")
    spk_emb = os.getenv("spk_emb", "face") #or "speech"

    # Experiment Configs
    batch_size = int(os.getenv("batch_size", 256)) #it was 256 #if gpu =4 -> 256%4 // For batch_size=64 no cuda out memory for gan
    add_blank = int(os.getenv("add_blank", 1))  # True: 1 / False: 0
    snet_emb = int(os.getenv("snet_emb", 1))  # True: 1 / False: 0
    n_spks = int(os.getenv("n_spks", 2007))  # libritts:247, lrs3: 2007
    multi_spks = int(os.getenv("multi_spks", 1))
    out_size = fix_len_compatibility(2 * sample_rate // 256)
    #model = os.getenv("model", "face-tts")
    gamma = os.getenv("gamma", 0.1)

    # Optimizer Configs
    optim_type = os.getenv("optim_type", "adam")
    schedule_type = os.getenv("schedule_type", "constant")
    learning_rate = float(os.getenv("learning_rate", 1e-4))
    end_lr = float(os.getenv("end_lr", 1e-7))
    weight_decay = float(os.getenv("weight_decay", 0.1))
    decay_power = float(os.getenv("decay_power", 1.0))
    max_steps = int(os.getenv("max_steps", 100000))

    save_step = int(os.getenv("save_step", 10000))
    warmup_steps = float(os.getenv("warmup_steps", 0))  # 1000

    video_data_root = os.getenv("video_data_root", "mp4")
    image_data_root = os.getenv("image_data_root", "jpg")
    audio_data_root = os.getenv("audio_data_root", "wav")

    #log_dir = os.getenv("CHECKPOINTS", "./logs")
    log_every_n_steps = int(os.getenv("log_every_n_steps", 1000))

    num_gpus = int(os.getenv("num_gpus", 4)) #it was 1 -> 
    per_gpu_batchsize = int(batch_size / num_gpus)
    num_nodes = int(os.getenv("num_nodes", 1))
    num_workers = int(os.getenv("num_workers", 2))  # Default is 2
    prefetch_factor = int(os.getenv("preftch_factor", 2))  # Default is 2
    # -----------------------------------------------------------------------------
    # GAN 
    # -----------------------------------------------------------------------------
    use_gan = int(os.getenv("use_gan", 1))  # 0 = False, 1 = True
    disc_base_channels = int(os.getenv("disc_base_channels", 32))
    disc_num_layers = int(os.getenv("disc_num_layers", 3))
    disc_lrelu_slope = float(os.getenv("disc_lrelu_slope", 0.2))
    disc_learning_rate = float(os.getenv("disc_learning_rate", 1e-6))
    lReLU_slope = float(os.getenv("lReLU_slope", 0.2))
    use_spectral_norm = int(os.getenv("use_spectral_norm", 0)) # True: 1 / False: 0
    residual_channels = int(os.getenv("residual_channels", 256))

    warmup_disc_epochs = int(os.getenv("warmup_disc_epochs", 10))
    freeze_gen_epochs = int(os.getenv("freeze_gen_epochs", 0))

    disc_loss_type = os.getenv("disc_loss_type", "bce")  # oder "mse", "hinge" 
    speaker_loss_weight = float(os.getenv("speaker_loss_weight", 0.01))
    lambda_adv = float(os.getenv("lambda_adv", 0.01))
    micro_batch_size = int(os.getenv("micro_batch_size", 16))


    # Inference Configs
    resume_from = os.getenv("resume_from", "./ckpts/facetts_lrs3.pt")
    test_txt = os.getenv("test_txt", "test/text.txt")
    use_custom = int(os.getenv("use_custom", 1))
    test_faceimg = os.getenv("test_faceimg", "test/face.png") #CFD-AF-200-228-N.
    timesteps = int(os.getenv("timesteps", 10))
    output_dir_orig = os.getenv("output_dir", "test/synth_voices_orig")
    output_dir_gan = os.getenv("output_dir", "test/synth_voices_gan_denoising")
    results_path = os.getenv("results_path", "evaluation")
    # SyncNet Configs
    syncnet_initw = float(os.getenv("syncnet_initw", 10.0))
    syncnet_initb = float(os.getenv("syncnet_initb", -5.0))

    #resume checkpoints from for inference
    infr_resume_from_orig = os.getenv("infr_resume_from_orig", "/mnt/qb/work/butz/bst080/faceGANtts/lightning_logs/facetts_original/checkpoints/last.ckpt")
    infr_resume_from_gan = os.getenv("infr_resume_from_gan", "/mnt/qb/work/butz/bst080/faceGANtts/lightning_logs/version_1176401/checkpoints/epoch=23-step=3956.ckpt") #/mnt/qb/work/butz/bst080/faceGANtts/lightning_logs/gan_fm_false/checkpoints/epoch=45-step=8004.ckpt

    val_check_interval = float(os.getenv("val_check_interval", 1.0))
    test_only = int(os.getenv("test_only", 0))