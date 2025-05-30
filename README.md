# SegAttnDetec
This is the official implementation of the paper:
  * [SegAttnDetec: A Segmentation-Aware Attention-Based Object Detector](https://www.sciencedirect.com/science/article/pii/S187705092501018X)


**Model Architecture**
---
![Model Arch](modelarch.svg)

**Training Stratergy**
---
![Training](threestagetraining.png)

**Experimental Results on the KITTI 2D Object Detection Dataset**
---
```latex
\begin{table}[ht]
    \centering
    \caption{Overall comparison of SegAttnDetec against key existing DL-based object detectors}
    \setlength{\tabcolsep}{4pt} % Default value: 6pt
    \begin{tabular}{|c|c|c|c|c|c|c|c|}
    \hline
        \multirow{2}{*}{Model} & \multirow{2}{*}{Backbone} & \multicolumn{3}{c|}{Category mAP (\%)} & \multirow{2}{*}{\vspace{.5cm}Overall} & \multirow{2}{*}{\vspace{.5cm}\# of Tr.} & \multirow{2}{*}{$\uparrow$ \%} \\ \cline{3-5}
        & & Easy & Moderate & Hard &  mAP (\%) & params &  \\ \hline\hline
        Faster R-CNN & VGG-16 (baseline) & 83.16 & 88.97 & 72.62 & 81.58 & - & - \\ \hline
        \multirow{2}{*}{Faster R-CNN} & \multirow{2}{*}{ResNet-50} & \multirow{2}{*}{83.08} & \multirow{2}{*}{79.28} & \multirow{2}{*}{73.71} & \multirow{2}{*}{78.69} & \multirow{2}{*}{41.1M} & \multirow{2}{*}{-3.5} \\
        & & & & & & & \\ \hline
        \multirow{2}{*}{Yolov5~\cite{yolo}} & \multirow{2}{*}{-} & \multirow{2}{*}{-} & \multirow{2}{*}{-} & \multirow{2}{*}{-} & \multirow{2}{*}{63.60} & \multirow{2}{*}{14.0M} & \multirow{2}{*}{-22.0} \\
        & & & & & & & \\ \hline
        \multirow{2}{*}{BiGA-YOLO~\cite{liu2023biga}} & \multirow{2}{*}{-} & \multirow{2}{*}{-} & \multirow{2}{*}{-} & \multirow{2}{*}{-} & \multirow{2}{*}{68.30} & \multirow{2}{*}{11.9M} & \multirow{2}{*}{-16.3} \\
        & & & & & & & \\ \hline
        SegAttnDetec & ResNet-50~(OD) + & \multirow{2}{*}{86.64} & \multirow{2}{*}{81.86} & \multirow{2}{*}{79.69} & \multirow{2}{*}{83.52} & \multirow{2}{*}{52.1M} & \multirow{2}{*}{+2.4} \\
        (proposed) & ResNet-101~(sem-seg) & & & & & & \\ \hline\hline
    \end{tabular}
    \label{tab:results}
    \begin{tablenotes}
        \small
        \item Note: $\uparrow$ \% - improvement \% compared to the baseline, \# of Tr. params - number of trainable parameters. 
    \end{tablenotes}
\end{table}

\begin{table}[ht]
\centering
\begin{threeparttable}
    \caption{Class-wise performance comparison wrt mAP \%}
    \setlength{\tabcolsep}{1pt} % Default value: 6pt
    \begin{tabular}{|c|c|c|c|c|c|c|}
    \hline 
    \multirow{2}{*}{\vspace{0.5cm}Object} & \multicolumn{2}{c|}{Easy Category} & \multicolumn{2}{c|}{Moderate Category} & \multicolumn{2}{c|}{Hard Category} \\ \cline{2-7}
    class & FR-CNN~\cite{gefanobject} & SegAttnDetec & FR-CNN~\cite{gefanobject} & SegAttnDetec & FR-CNN~\cite{gefanobject} & SegAttnDetec \\ \hline\hline
    Car & 84.81 & 97.22 & 86.18 & 89.70 & 78.03 & 88.90 \\ \hline
    Pedestrian & 76.52 & 79.96 & 59.98 & 73.92 & 51.84 & 66.62 \\ \hline
    Cyclist & 74.72 & 78.59 & 56.83 & 70.50 & 49.60 & 68.86 \\ \hline\hline
    \end{tabular}
    \label{tab:benchmark_comparison}
    \end{threeparttable}
\end{table}
```

**Environment** 
---
```bash
torch==2.1.2
torchvision==0.16.2
lightning==2.4.0
lightning_utilities==0.11.8
```
