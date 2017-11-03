from matplotlib import pyplot as plt

stat_files = [
    ('Libre', [
        'stats_dtlz1_3.txt',
        ]),
    ('NSGA2', [
        'stats_dtlz1_3__nsga2_1.txt',
        'stats_dtlz1_3__nsga2_2.txt',
        'stats_dtlz1_3__nsga2_3.txt',
    ]),
]

# Get color
def get_color():
    for color in ['b', 'g', 'r']:
        yield color
clrs = get_color()


def show_stats_for_one_file(alg, stats):  # evals, hv, uni
    plt.plot([e[0] for e in stats], [e[1] for e in stats], clrs.next() + '-', label=alg, linewidth=2)

def show_stats_for_files(alg, stats):  # (evals, [hvs], [uni])
    clr = clrs.next()
    plt.plot([e[0] for e in stats], [min(e[1]) for e in stats], clr + '-', label=alg, linewidth=2)
    plt.plot([e[0] for e in stats], [sum(e[1])/len(e[1]) for e in stats], clr + '-', label=alg)
    plt.plot([e[0] for e in stats], [max(e[1]) for e in stats], clr + '-', label=alg, linewidth=2)
    plt.fill_between([e[0] for e in stats], [min(e[1]) for e in stats], [max(e[1]) for e in stats], color=clr, alpha='0.3')

def parse_stats_for_one_file(files):
    open_f = open(files[0], 'r')
    stats = []
    for line in open_f:
        stats.append([float(e) for e in line.split()])
    open_f.close()
    return stats

def parse_stats_for_files(fs):
    stats = []     # [ev, hv, uni]  [ev, hv, uni]  [ev, hv, uni]
    for f in fs:
        stats.append(parse_stats_for_one_file([f]))
    merged = []
    for i in range(len(stats[0])):
        merged.append((stats[0][i][0], [s[i][1] for s in stats], [s[i][2] for s in stats]))
    return merged

if __name__ == '__main__':
    for alg, files in stat_files:
        print('Showing', alg)
        if len(files) == 1:
            show_stats_for_one_file(alg, parse_stats_for_one_file(files))
        else:
            show_stats_for_files(alg, parse_stats_for_files(files))
    plt.ylabel('Hyper volume')
    plt.xlabel('Trials')
    plt.show()
