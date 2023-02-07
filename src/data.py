import torch
from torch_geometric.datasets import IMDB, LastFM, MovieLens
from torch_geometric.transforms import AddMetaPaths
from torch_geometric.utils import remove_self_loops
from pytorch_lightning import seed_everything
from dataset import *

def load_dataset_pyg(dataname="HGB_ACM", seed=0, device=None):
    
    seed_everything(seed)
    
    if dataname[:3] == "HGB":
        
        dataset = HGBDataset(root="../datasets/hgb", name=dataname[4:])
        
        if dataset.name == "acm":
                
            graph = dataset.data
            
            # print(graph["paper"].y[graph["paper"].test_mask][:100])

            graph["paper", "self", "paper"].edge_index = torch.arange(len(graph["paper"].x))[None, :].repeat(2, 1)
            
            metapaths = [
                [("paper", "cite", "paper"),  ("paper", "self", "paper")],
                # [("paper", "ref", "paper"),   ("paper", "self", "paper")],
                # [("paper", "to", "author")],
                # [("paper", "cite", "paper") ,   ("paper",   "ref", "paper")],
                # [("paper", "cite", "paper"),   ("paper", "to", "author"),  ("author",  "to", "paper")],
               #  [("paper", "cite", "paper"),   ("paper", "to", "subject"), ("subject", "to", "paper")],
                # [("paper", "ref", "paper"),   ("paper", "to", "author"),  ("author",  "to", "paper")],
                # [("paper", "ref", "paper"),   ("paper", "to", "subject"), ("subject", "to", "paper")],
                # [("paper", "cite", "paper"),   ("paper", "to", "term"),    ("term",    "to", "paper")],
                [("paper", "to",   "author"),  ("author",  "to", "paper")],
                [("paper", "to",   "subject"), ("subject", "to", "paper")],
                [("paper", "to",   "term"),    ("term",    "to", "paper")]
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            target = "paper"

        elif dataset.name == "dblp":
            
            graph = dataset.data
            
            metapaths = [
              [("author", "to", "paper"), ("paper", "to", "author")],
               [("author", "to", "paper"), ("paper",   "to", "term"), ("term", "to", "paper"), ("paper", "to", "author")],
               [("author", "to", "paper"), ("paper",   "to", "venue"), ("venue", "to", "paper"), ("paper", "to", "author")]
            ]
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            #
            # print(graph)

            target = "author"
            
        elif dataset.name == "imdb":
            
            target = "movie"
            
            graph = dataset.data
            
            graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)

            metapaths = [[("movie", "to", "director"),   ("director", "to", "movie")], [("movie", ">actorh", "actor"), ("actor", "to", "movie")]]
                # [("movie", "to", "director"),   ("director", "to", "movie"), ("movie", ">actorh", "actor"), ("actor", "to", "movie")],
                # [("movie", "to", "director"),   ("director", "to", "movie"), ("movie", "to", "keyword"),    ("keyword", "to", "movie")],
                # [("movie", ">actorh", "actor"), ("actor", "to", "movie"),    ("movie", "to", "keyword"),    ("keyword", "to", "movie")]
            
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            
            # print(graph)
            
        elif dataset.name == "freebase":
            
            graph = dataset.data
            graph["book", "self", "book"].edge_index = torch.arange(graph["book"].num_nodes)[None, :].repeat(2, 1)
            edge_types = graph.edge_types
            
            for etype in edge_types:
                graph[etype[2], "rev_"+etype[1], etype[0]].edge_index = graph[etype].edge_index[torch.LongTensor([1, 0])]
            
            metapaths = [
                [("book", "and", "book"), ("book", "self", "book")]]
                # [("book", "rev_about", "business"),  ("business", "about", "book")],
                # [("book", "to", "film"),  ("film", "rev_to", "book")],
                # [("book", "on", "location"),  ("location", "rev_on", "music"), ("music", "in", "book")],
                # [("book", "about", "organization"),  ("organization", "in", "film"), ("film", "rev_to", "book")],
                # [("book", "rev_to", "people"), ("people", "to", "book")],
                # [("book", "rev_to", "people"), ("people", "to", "sports"), ("sports", "rev_on", "book")],
            
            
            graph = AddMetaPaths(metapaths=metapaths)(graph)
            
            # print(graph)

            target = "book"
            
        # graph = graph.to(device)
        
        print(graph.node_types)
        print(graph.edge_types)

        if dataset.name != "freebase":
            features = graph[target].x
        else:
            dim = graph[target].num_nodes
            print(dim)
            indices  = np.vstack((np.arange(dim), np.arange(dim)))
            indices  = torch.LongTensor(indices)
            values   = torch.FloatTensor(np.ones(dim))
            # print(features_list[i].shape)
            features = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
            
    elif dataname[:2] == "FB":
        
        target = "person"
        
        dataset = FBDataset(root="../datasets/Facebook", name=dataname[3:])
        
        graph = dataset.data

        print(graph)
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        metapaths = [[("person", "to", "person"), ("person", "self", "person")],
                     [("person", "to", "high school"),  ("high school", "rev_to", "person")],
                    [("person", "to", "major"),  ("major", "rev_to", "person")]
                    ]
    
        graph = AddMetaPaths(metapaths=metapaths)(graph)
    
    elif dataname == "Actor":
        
        target = "starring"
        
        dataset = Actor(root="../datasets/")
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        metapaths = [[("starring", "rel_0", "starring"), ("starring", "self",      "starring")],
                    [("starring",  "rel_2",   "writer"), ("writer",   "rel_2_rev", "starring")],
                    [("starring",  "rel_1", "director"), ("director", "rel_1_rev", "starring")]]
        
        #[[("starring", "rel_2",   "writer"), ("writer",   "rel_2_rev", "starring")],
                    #  [("starring", "rel_0", "starring"), ("starring", "self",      "starring")],
                     #[("starring", "rel_1", "director"), ("director", "rel_1_rev", "starring")]]
                    
        

        graph = AddMetaPaths(metapaths=metapaths)(graph)
        
    elif dataname[:4] == "Fake":
        
        target = "news"
        
        dataset = FakeNewsNet(root="../datasets/", name=dataname[5:])
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
    elif dataname == "Liar":
        
        target = "news"
        
        dataset = Liar(root="../datasets/")
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        metapaths = [[("news", "to", "speaker"), ("speaker", "to", "news")],
                     # [("news", "to", "subject"), ("subject", "to", "news")],
                     [("news", "to", "context"), ("context", "to", "news")],
                   ]
        
        # [[("news", "to", "context"), ("context", "to", "news")]]
                    #  [("news", "to", "speaker"), ("speaker", "to", "news")],
                    #  [("news", "to", "subject"), ("subject", "to", "news")]]
        
        graph = AddMetaPaths(metapaths=metapaths)(graph)
    
    elif dataname[:2] == "FD":
        
        target = "user"
        
        dataset = Fraud(root="../datasets/", name=dataname[3:])
        
        graph = dataset.data
        
        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)
        
        if dataname[3:] == "Amazon":
            metapaths = [[("user", "p", "user"), ("user", "self", "user")]]
                         # [("user", "to", "speaker"), ("speaker", "to", "news")]]
                         
        graph = AddMetaPaths(metapaths=metapaths)(graph)

    elif dataname == "Patent":

        target  = "patent"

        dataset = Patent(root="../datasets/", year=1998)

        graph   = dataset.data

        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)

        metapaths = [[("patent", "citing", "patent"), ("patent", "self", "patent")],
                     [("patent", "cited", "patent"), ("patent", "self", "patent")],
                     [("patent", "to", "inventer"), ("inventer", "to", "patent")]]
                    #  [("patent", "to", "country"), ("country", "to", "patent")]]

        graph = AddMetaPaths(metapaths=metapaths)(graph)

        edge_index = graph[("patent", "citing", "patent")].edge_index

        edge_index = remove_self_loops(edge_index)[0]

        print(graph)

    elif dataname == "IMDB":

        target = "movie"

        dataset = IMDB(root="../datasets/")

        graph = dataset.data

        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)

        print(graph)

        metapaths = [[("movie", "to", "director"), ("director", "to", "movie")],
                    [("movie", "to", "actor"), ("actor", "to", "movie")],
        ]
    
        graph = AddMetaPaths(metapaths=metapaths)(graph)

    elif dataname == "MovieLens":

        target = "movie"

        dataset = MovieLens(root="../datasets/")

        graph = dataset.data

        graph[target, "self", target].edge_index = torch.arange(len(graph[target].x))[None, :].repeat(2, 1)

        print(graph)

        metapaths = [[("movie", "to", "director"), ("director", "to", "movie")],
                     [("movie", "to", "actor"), ("actor", "to", "movie")]]
    
        graph = AddMetaPaths(metapaths=metapaths)(graph)

    print(graph[target].y.shape)
    labels      = graph[target].y  
    
    if dataname != IMDB:
        train_index = graph[target].train_mask.nonzero()
        val_index   = train_index[torch.randperm(len(train_index))[:len(train_index) // 4]]
        val_mask    = torch.zeros(graph[target].num_nodes)
        val_mask[val_index] = 1
        val_mask    = val_mask.bool()

        train_mask  = graph[target].train_mask
        train_mask[val_mask] = 0

        train_index = train_mask.to(device)
        val_index   = val_mask.to(device)
        test_index  = graph[target].test_mask
        test_mask   = test_index
    else:
        train_index = graph[target].train_mask.nonzero()
        val_index   = graph[target].val_mask.nonzero()
        test_index  = graph[target].test_mask.nonzero()
        
    num_labels = labels.max().item()+1 if len(labels.size()) == 1 else labels.shape[1]

    return graph, target, train_mask, val_mask, test_mask, labels, num_labels