from TransferModel.DataUtils.DataProcessor import featurize_seqs_host
from TransferModel.Models.BilstmTargetModel import BiLSTMTargetModel


def err_model(name):
    raise ValueError('Model {} not supported'.format(name))


def get_target_model(args, parentModel, seq_len, vocab_size,
                     inference_batch_size=200):
    if args.model_name == 'bilstm':
        model = BiLSTMTargetModel(
            seq_len,
            parentModel,
            vocab_size,
            target_size=len(args.targetNames),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.dim,
            n_hidden=args.n_hidden,
            n_epochs=args.n_epochs,
            batch_size=args.batch_size,
            inference_batch_size=args.inf_batch_size,
            cache_dir='target/{}'.format(args.namespace),
            seed=args.seed,
            verbose=True,
        )
    else:
        err_model(args.model_name)

    return model


def fit_model_host(model, seqs, vocabulary, labelVocab):
    X, lengths = featurize_seqs_host(seqs, vocabulary)
    y = seqs['target']
    model.fit(X, lengths, y)
    return model
