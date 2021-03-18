import argparse
from time import time
from datetime import datetime

from models import *
from utils import *


def parse_sampled_batch(batch, args):
    uid, seq, pos, neg, nbr, nbr_iid = batch
    uid = uid.long()
    seq = seq.long()[:, -args.seq_maxlen:]
    pos = pos.long()[:, -args.seq_maxlen:]
    neg = neg.long()[:, -args.seq_maxlen:]
    nbr = nbr.long()[:, :args.nbr_maxlen]
    nbr_iid = nbr_iid.long()[:, :args.nbr_maxlen, -args.seq_maxlen:]
    batch = [uid, seq, pos, neg, nbr, nbr_iid]
    indices = torch.where(pos != 0)
    return batch, indices


def train(model, opt, shdlr, train_loader, args):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        parsed_batch, indices = parse_sampled_batch(batch, args)

        opt.zero_grad()
        shdlr.step()
        pos_logits, neg_logits, user_emb, pos_hi, neg_hi = model(parsed_batch)

        # Label Loss
        loss = 0.0
        if args.loss_type == 'bce':
            loss += F.binary_cross_entropy_with_logits(pos_logits[indices], torch.ones_like(pos_logits)[indices]) + \
                    F.binary_cross_entropy_with_logits(neg_logits[indices], torch.zeros_like(neg_logits)[indices])
        elif args.loss_type == 'bpr':
            loss += F.softplus(neg_logits[indices] - pos_logits[indices]).mean()

        # Embedding Reg loss
        user_norm = user_emb.norm(2, dim=-1).pow(2).mean()
        item_norm = pos_hi.norm(2, dim=-1).pow(2).mean() + neg_hi.norm(2, dim=-1).pow(2).mean()
        emb_reg_loss = args.emb_reg * (user_norm + item_norm) / 2
        loss += emb_reg_loss

        loss.backward()
        opt.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)




def main():
    parser = argparse.ArgumentParser(description='BPRMF')
    parser.add_argument('--model', default='BPRMF')
    parser.add_argument('--dataset', default='Yelp')

    # Model Config
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50, help='fixed, or change with sampled train_batches')
    parser.add_argument('--nbr_maxlen', type=int, default=20, help='fixed, or change with sampled train_batches')

    # Train Config
    parser.add_argument('--batch_size', type=int, default=1024, help='fixed, or change with sampled train_batches')
    parser.add_argument('--droprate', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.75)
    parser.add_argument('--l2rg', type=float, default=0.0)
    parser.add_argument('--emb_reg', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--check_epoch', type=int, default=1)
    parser.add_argument('--loss_type', default='bpr', help='bce/bpr')
    parser.add_argument('--patience', type=int, default=5)

    # Something else
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--test_time', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=False)
    args = parser.parse_args()

    user_num = 270770 + 1
    item_num = 184134 + 1


    print('Loading...')
    st = time()
    train_loader, val_loader, test_loader, user_train, eval_users = load_ds(args, item_num)
    print('Loaded Yelp dataset with {} users {} items in {:.2f}s'.format(user_num, item_num, time()-st))

    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'../saved_models/{args.model}_{args.dataset}_{timestr}.pth'
    logger = get_logger(f'../logs/{args.model}_{args.dataset}_{timestr}.log')
    logger.info(args)
    device = torch.device(args.device)

    metrics_list = []
    for r in range(args.repeat):
        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = BPRMF(user_num, item_num, args)
        model = model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2rg)
        lr_scheduler = StepwiseLR(opt, init_lr=args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decay_rate)
        best_score = patience_cnt = 0

        for epoch in range(1, args.max_epochs):
            st = time()
            train_loss = train(model, opt, lr_scheduler, train_loader, args)
            print('Epoch:{} Train Loss={:.4f} Time={:.2f}s LR={:.8f}'.format(
                epoch, train_loss, time()-st, lr_scheduler.get_lr()))

            if epoch % args.check_epoch == 0 and epoch >= 0:
                val_metrics = evaluate(model, val_loader, eval_users)
                hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = val_metrics
                logger.info(
                    'Iter={} Epoch={:04d} Val HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                    .format(r, epoch, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))

                if best_score < hr10:
                    torch.save(model.state_dict(), model_path)
                    print('Validation score increased: {:.4f} --> {:.4f}'.format(best_score, hr10))
                    best_score = hr10
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt == args.patience:
                    print('Early Stop!!!')
                    break

        print('Testing', end='')
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate(model, test_loader, eval_users)
        hr5, hr10, hr20, ndcg5, ndcg10, ndcg20 = test_metrics
        logger.info('Iter={} Tst HR(5/10/20)={:.4f}/{:.4f}/{:.4f} NDCG(5/10/20)={:.4f}/{:.4f}/{:.4f}'
                    .format(r, hr5, hr10, hr20, ndcg5, ndcg10, ndcg20))
        metrics_list.append(test_metrics)

    metrics = np.array(metrics_list)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)
    print('Test Summary:')
    logger.info('Mean hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        means[0], means[1], means[2], means[3], means[4], means[5]))
    logger.info('Std  hr5={:.4f}, hr10={:.4f}, hr20={:.4f}, ndcg5={:.4f}, ndcg10={:.4f}, ndcg20={:.4f}'.format(
        stds[0], stds[1], stds[2], stds[3], stds[4], stds[5]))
    logger.info("Done")


if __name__ == '__main__':
    main()
