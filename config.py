config = {
    # training files
    'url_root': 'https://m.media-amazon.com/images/I/',
    'data_root': './data',
    'gallery_csv': 'train_gallery_file.csv',
    'train_csv': 'apparel_train_annotation.csv',
    'dev_jsonl': 'query_file_released.jsonl',
    'test_csv': '',
    'train_img': 'data/train_images',
    'dev_img': 'data/dev_images',
    'test_img': '',
    'train_qnet_emb': './data/clip_bknd/train_qnet_emb.pt',
    'train_cnet_emb': './data/clip_bknd/train_cnet_emb.pt',
    'cap_datapoints': True,
    'max_datapoints': 100,
    'read_from_folder': False,

    ## hyperparameters
    'batch_size': 64,
    'num_workers': 4,
    'random_seed': 1,
    'learning_rate': 0.001,
    'shuffle_data': False,
    'split_train_into_val': True,
    'validation_split': 0.2,
    'epochs': 50,

    ## separator for text
    'text_sep': ' and ',
    # 'text_sep': ' ',

    ## matplotlib loss curve file
    'clip_crs_em_loss_img': 'clip_crs_em_loss.png',

    ## clip backend
    'clip_backend': 'ViT-B/32',
    # 'clip_backend': 'ViT-L/14',

    ## inference
    'use_clip_aprch_inf': True,
    'use_resnet_aprch_inf': False,
    'clip_bknd_cnet_model_path': '',
    'clip_bknd_qnet_model_path': '',
    'gal_imgs_ftrs_path': '',
    'qry_ftrs_path': '',
    'dev_phase_res_file': 'data/dev_phase_results/dev_phase_results.jsonl',

    ## downloading images
    'dev_images': 'data/dev_images',

}