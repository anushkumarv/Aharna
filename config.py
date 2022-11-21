config = {
    # training files
    'url_root': 'https://m.media-amazon.com/images/I/',
    'data_root': './data',
    'train_csv': 'apparel_train_annotation.csv',
    'dev_jsonl': 'query_file_released.jsonl',
    'test_csv': '',
    'train_img': '',
    'dev_img': '',
    'test_img': '',

    # dataloaders
    'batch_size': 64,
    'num_workers': 4,

    ## separator for text
    'text_sep': ' and ',
    # 'text_sep': ' ',

    ## clip backend
    'clip_backend': 'ViT-B/32',
    # 'clip_backend': 'ViT-L/14',
}