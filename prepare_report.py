"""Audit existing runs and render the report; never train or choose a model on test."""
import argparse
import json
import shlex
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORDER = ['f1036b', '0a3948', 'c591f3', '9279f2', '646c9d', '4822d8', 'cd08a0']
SHA = '10ac1fb033e3b1845bc52ea2e2229c2982ad11f0'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/petr/optimizers'))
    parser.add_argument('--out', type=Path, default=Path('reports/resnet18'))
    parser.add_argument('--print-training-commands', action='store_true', help='Print the original settings; do not execute training')
    args = parser.parse_args()
    if args.print_training_commands:
        settings = [
            (10,'sgd',1e-4,.2,0), (10,'adamw',1e-4,.2,0),
            (10,'adamw',1e-4,.4,0), (7,'adamw',1e-4,.1,0),
            (7,'adamw',1e-4,.1,.01), (7,'adamw',1e-4,.1,.05),
            (7,'adamw',5e-5,.1,.01),
        ]
        for epochs, optimizer, lr, dropout, wd in settings:
            cmd = ['uv','run','iml-train','--config','app/models/configs/resnet18.json',
                   '--data','data/training_dataset_v2','--member','petr','--topic','optimizers']
            for key, value in dict(epochs=epochs, optimizer=optimizer, lr=lr, dropout=dropout,
                                  weight_decay=wd, pretrained='true', checkpoint_metric='eer',
                                  early_stopping_patience=0, batch_size=32, seed=42).items():
                cmd.extend(['--set',f'{key}={value}'])
            print(shlex.join(cmd))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'plots').mkdir(exist_ok=True)
    records = []
    for i, suffix in enumerate(ORDER, 1):
        matches = list(args.runs.glob('*_' + suffix))
        if len(matches) != 1:
            raise ValueError(f'Expected one existing run ending in {suffix}')
        d = matches[0]
        c = json.loads((d/'config.json').read_text())
        m = json.loads((d/'metrics.json').read_text())
        data = json.loads((d/'data.json').read_text())
        h = pd.read_csv(d/'history.csv')
        assert data['sha1'] == SHA and c['pretrained'] is True
        assert c['checkpoint_metric'] == 'eer' and m['select_metric'] == 'eer'
        assert c['seed'] == 42 and c['batch_size'] == 32
        assert len(h) == c['epochs'] == m['epochs_run']
        assert int(h.loc[h.val_eer.idxmin(), 'epoch']) == m['best_epoch']
        counts = {}
        for split in ['train', 'val', 'test']:
            with np.load(d/f'predictions_{split}.npz') as a:
                y, s = a['labels'], a['scores']
            assert np.isfinite(s).all()
            assert len(y) == m[split]['n']
            pos, neg = s[y == 1], s[y == 0]
            threshold = m['decision_threshold']
            fp, fn = int((neg >= threshold).sum()), int((pos < threshold).sum())
            for key, value in {'accuracy':1-(fp+fn)/len(y), 'far':fp/len(neg), 'frr':fn/len(pos)}.items():
                assert np.isclose(value, m[split][key], atol=1e-12), (d, split, key)
            thresholds = np.r_[np.inf, np.unique(s)[::-1]]
            fps = (neg[:,None] >= thresholds).sum(0)
            fns = (pos[:,None] < thresholds).sum(0)
            distances = abs(fps*len(pos) - fns*len(neg))
            rates = (fps/len(neg) + fns/len(pos))/2
            assert np.any(np.isclose(rates[distances == distances.min()], m[split]['eer'], atol=1e-12))
            if split != 'train':
                assert len(y) == 63 and len(pos) == 28 and len(neg) == 35
            counts[split] = {'false_accepts':fp, 'false_rejects':fn}
        records.append(dict(id=f'R{i}', run=d.name, config=c, metrics=m, errors=counts, history=h))
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    for name, rs, title in [
        ('optimizers_dropout', records[:3], 'ResNet-18 fine-tuning: optimizer and dropout trials (10 epochs)'),
        ('regularization_lr', records[3:], 'ResNet-18 fine-tuning: regularization and learning-rate trials (7 epochs)'),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        handles, labels = [], []
        for r, color, marker in zip(rs, ['#0072b2','#d55e00','#009e73','#aa4499'], ['o','s','^','D']):
            h, c, m = r['history'], r['config'], r['metrics']
            for ax, col, scale in zip(axes, ['train_loss','val_loss','val_eer'], [1,1,100]):
                line, = ax.plot(h.epoch, h[col]*scale, color=color, marker=marker, ms=4, lw=1.7)
                best = h.loc[h.epoch == m['best_epoch']].iloc[0]
                ax.scatter(best.epoch, best[col]*scale, s=120, facecolors='none', edgecolors=color, linewidths=1.6, zorder=5)
            handles.append(line)
            labels.append(f"{r['id']}: {c['optimizer'].upper()}, LR={c['lr']:g}, dropout={c['dropout']:g}, WD={c['weight_decay']:g}; saved epoch {m['best_epoch']}")
        losses = np.concatenate([r['history'][['train_loss','val_loss']].to_numpy().ravel() for r in rs])
        for ax in axes[:2]:
            ax.set_yscale('log')
            ax.set_ylim(losses.min()/1.5, losses.max()*1.5)
            ax.set_ylabel('Cross-entropy (log scale)')
        for ax, t in zip(axes, ['Training loss', 'Validation loss', 'Validation EER']):
            ax.set_title(t)
            ax.set_xlabel('Epoch')
            ax.set_xticks(range(1, rs[0]['config']['epochs']+1))
            ax.grid(alpha=.22, which='major')
        axes[2].set_ylabel('EER (%)')
        axes[2].set_ylim(-2, max(r['history'].val_eer.max()*100 for r in rs)+3)
        fig.suptitle(title, fontsize=15, y=.98)
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.005), ncol=1, fontsize=10, frameon=False)
        fig.subplots_adjust(left=.065,right=.985,top=.86,bottom=.30,wspace=.30)
        fig.savefig(args.out/'plots'/f'{name}.png',dpi=180,facecolor='white')
        plt.close(fig)
    serial = [{k:v for k,v in r.items() if k != 'history'} for r in records]
    (args.out/'verified_runs.json').write_text(json.dumps(serial,indent=2)+'\n')
    for r in records:
        m=r['metrics']; print(r['id'],r['run'], 'best',m['best_epoch'],'test errors',r['errors']['test'])

if __name__ == '__main__':
    main()
