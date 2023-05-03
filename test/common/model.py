import torch
from torch import nn
from hgl import mp
from typing import Union
import math


# adapted from HGTLayerHetero in [[third_party/OthersArtifacts/graphiler/examples/HGT/HGT_DGL.py]]
class HGTLayerHetero(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads=1,
        dropout=0.2,
        use_norm=False,
    ):
        super(HGTLayerHetero, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, G: mp.HeteroGraph, h):
        
        node_dict, edge_dict = self.node_dict, self.edge_dict
        for (srctype, etype, dsttype), sub_graph in G:
            

            k_linear = self.k_linears[node_dict[srctype]]
            v_linear = self.v_linears[node_dict[srctype]]
            q_linear = self.q_linears[node_dict[dsttype]]
            if len(node_dict) == 1:
                k = k_linear(h).view(-1, self.n_heads, self.d_k)
                v = v_linear(h).view(-1, self.n_heads, self.d_k)
                q = q_linear(h).view(-1, self.n_heads, self.d_k)
            else:
                k = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                v = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                q = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

            e_id = self.edge_dict[(srctype, etype, dsttype)]

            relation_att = self.relation_att[e_id]
            relation_pri = self.relation_pri[e_id]
            relation_msg = self.relation_msg[e_id]

            k = torch.einsum("bij,ijk->bik", k, relation_att)
            v = torch.einsum("bij,ijk->bik", v, relation_msg)

            sub_graph.src_node["k"] = k
            sub_graph.dst_node["q"] = q
            sub_graph.src_node["v_%d" % e_id] = v

            sub_graph.apply_edges(mp.Fn.v_dot_u("q", "k", "t"))
            attn_score = (
                sub_graph.edata.pop("t").sum(-1) * relation_pri / self.sqrt_dk
            )
            attn_score = edge_softmax(sub_graph, attn_score, norm_by="dst")

            sub_graph.edata["t"] = attn_score.unsqueeze(-1)

        G.multi_update_all(
            {
                etype: (fn.u_mul_e("v_%d" % e_id, "t", "m"), fn.sum("m", "t"))
                for etype, e_id in edge_dict.items()
            },
            cross_reducer="mean",
        )

        new_h = {}
        for ntype in G.ntypes:
            """
            Step 3: Target-specific Aggregation
            x = norm( W[node_type] * gelu( Agg(x) ) + x )
            """
            n_id = node_dict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            if "t" not in G.nodes[ntype].data:
                continue
            t = G.nodes[ntype].data["t"].view(-1, self.out_dim)
            trans_out = self.drop(self.a_linears[n_id](t))
            trans_out = trans_out * alpha  # + h[ntype] * (1-alpha) ?
            if self.use_norm:
                new_h[ntype] = self.norms[n_id](trans_out)
            else:
                new_h[ntype] = trans_out
        return new_h


class GCNLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.fc = nn.Linear(in_features, out_features)

    def forward(self,
                graph: mp.Graph,
                norm: torch.Tensor,
                x: Union[torch.Tensor, list]):
        if isinstance(x, (list, tuple)):
            assert len(x) == 2
            x = x[0]
        h = self.fc(x)
        h = h.view(size=[
            -1, 1, h.size(-1)
        ])
        graph.src_node['u'] = h
        graph.message_func(mp.Fn.copy_u('u', 'm'))
        graph.reduce_func(mp.Fn.aggregate_sum('m', 'v'))
        h = torch.squeeze(graph.dst_node['v'], dim=1)
        return torch.multiply(norm, h)


class GCNModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        self.i2h = GCNLayer(
            in_features, gnn_features
        )
        self.h2o = GCNLayer(
            gnn_features, out_features
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU(),
        )

    def forward(self,
                graph: mp.Graph,
                x: torch.Tensor,
                norm: torch.Tensor):
        h = self.i2h(graph, norm, x)
        h = self.activation(h)
        h = self.h2o(graph, norm, h)
        return h


class GATLayer(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_heads: int):
        nn.Module.__init__(self)
        #
        self.n_heads = n_heads
        self.n_features = out_features
        self.linear_q = nn.Linear(n_heads * out_features, n_heads)
        self.linear_k = nn.Linear(n_heads * out_features, n_heads)
        self.linear_v = nn.Linear(in_features, n_heads * out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, graph: mp.Graph, x):
        # different to bert
        if isinstance(x, (list, tuple)):
            assert len(x) == 2
            src_size = x[0].size(0)
            dst_size = x[1].size(0)
            graph.blk.size[0] == src_size
            graph.blk.size[1] == dst_size
            #
            h_src = self.linear_v(x[0])
            h_dst = self.linear_v(x[1])
        elif isinstance(x, torch.Tensor):
            h_src = h_dst = self.linear_v(x)
        else:
            raise TypeError
        q = self.linear_q(h_dst)
        k = self.linear_k(h_src)
        h_src = h_src.view(size=[
            -1, self.n_heads,
            self.n_features
        ])
        graph.src_node['q'] = q
        graph.dst_node['k'] = k
        graph.src_node['u'] = h_src

        # gat attention
        graph.message_func(mp.Fn.u_add_v('k', 'q', 'e'))
        graph.edge['coeff'] = self.leaky_relu(graph.edge['e'])
        graph.message_func(mp.Fn.edge_softmax('coeff', 'attn'))
        graph.message_func(mp.Fn.u_mul_e('u', 'attn', 'm'))
        graph.reduce_func(mp.Fn.aggregate_sum('m', 'v'))
        return torch.mean(graph.dst_node['v'], dim=1)


class GATModel(nn.Module):
    def __init__(self,
                 in_features: int,
                 gnn_features: int,
                 out_features: int,
                 n_heads: int = 8):
        nn.Module.__init__(self)
        #
        self.i2h = GATLayer(
            in_features, gnn_features,
            n_heads=n_heads
        )
        self.h2o = GATLayer(
            gnn_features, out_features,
            n_heads=n_heads
        )
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self, graph: mp.Graph, x: torch.Tensor):
        h = self.i2h(graph, x)
        h = self.activation(h)
        h = self.h2o(graph, h)
        return h


class HeteroGraphConv(nn.Module):
    def __init__(self, convs: dict):
        nn.Module.__init__(self)
        self.convs = nn.ModuleDict(convs)

    def forward(self,
                hgraph: mp.HeteroGraph,
                xs: dict,
                norms: dict = None):
        counts = {}
        outputs = {}
        for (sty, ety, dty), graph \
                in hgraph:
            if norms:
                res = self.convs[ety](
                    graph,
                    norms[str((sty, ety, dty))],
                    (xs[sty], xs[dty])
                )
            else:
                res = self.convs[ety](
                    graph, (xs[sty], xs[dty])
                )
            if dty not in counts:
                counts[dty] = 1
            else:
                counts[dty] += 1
            if dty not in outputs:
                outputs[dty] = res
            else:
                exist = outputs[dty]
                outputs[dty] = exist + res
        for dty, res in outputs.items():
            outputs[dty] = res / counts[dty]
        return outputs


class REmbedding(nn.Module):
    def __init__(self,
                 hgraph: mp.HeteroGraph,
                 embedding_dim: int):
        nn.Module.__init__(self)
        self.embeds = nn.ModuleDict({
            nty: nn.Embedding(
                num, embedding_dim
            )
            for nty, num in hgraph.nty2num.items()
        })

    def forward(self,
                hgraph: mp.HeteroGraph,
                xs: dict):
        return {
            nty: self.embeds[
                nty
            ](xs[nty])
            for nty in hgraph.nty2num
        }


class RGCNModel(nn.Module):
    def __init__(self,
                 hgraph: mp.HeteroGraph,
                 in_features: int,
                 #gnn_features: int,
                 out_features: int):
        nn.Module.__init__(self)
        #
        # self.em = REmbedding(
        #     hgraph=hgraph,
        #     embedding_dim=in_features
        # )
        self.i2h = HeteroGraphConv({
            ety: GCNLayer(
                in_features=in_features,
                out_features=out_features
            )
            for ety in hgraph.etypes
        })
        #self.h2o = HeteroGraphConv({
        #    ety: GCNLayer(
        #        in_features=gnn_features,
        #        out_features=out_features
        #    )
        #    for ety in hgraph.etypes
        #})
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ReLU()
        )

    def forward(self,
                hgraph: mp.HeteroGraph,
                norms: dict, xs: dict):
        #xs = self.em(hgraph, xs)
        hs = self.i2h(hgraph, xs, norms)
        hs = {
            k: self.activation(h)
            for k, h in hs.items()
        }
        #hs = self.h2o(hgraph, hs, norms)
        return hs


class RGATModel(nn.Module):
    def __init__(self,
                 hgraph: mp.HeteroGraph,
                 in_features: int,
                 #gnn_features: int,
                 out_features: int,
                 n_heads: int = 1):
        nn.Module.__init__(self)
        #
        # self.em = REmbedding(
        #     hgraph=hgraph,
        #     embedding_dim=in_features
        # )
        self.i2h = HeteroGraphConv({
            ety: GATLayer(
                in_features=in_features,
                out_features=out_features,
                n_heads=n_heads
            )
            for ety in hgraph.etypes
        })
        #self.h2o = HeteroGraphConv({
        #    ety: GATLayer(
        #        in_features=gnn_features,
        #        out_features=out_features,
        #        n_heads=n_heads
        #    )
        #    for ety in hgraph.etypes
        #})
        self.activation = nn.Sequential(
            nn.Dropout(0.5),
            nn.ELU()
        )

    def forward(self,
                hgraph: mp.HeteroGraph,
                xs: dict):
        # xs = self.em(hgraph, xs)
        hs = self.i2h(hgraph, xs)
        hs = {
            k: self.activation(h)
            for k, h in hs.items()
        }
        #hs = self.h2o(hgraph, hs)
        return hs
