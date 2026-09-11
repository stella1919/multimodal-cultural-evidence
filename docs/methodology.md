# Methodology

## Data collection and cleaning
Records are retrieved from The Metropolitan Museum of Art Collection API, cached by object ID, de-duplicated, and retained with source URLs and the museum's public-domain flag. Missing values remain missing. A pilot record is prioritized when it has a title, accessible image and object-page URLs, and usable contextual metadata.

## Context categorization
`cultural_context` is a researcher-derived analytical variable. Geography, country, and region receive higher weight than incidental title terms. Tibet/Himalaya/Nepal are assigned to Himalayan; China/Japan/Korea and selected period terms to East_Asia; India/South Asia and related geography to South_Asia. Ambiguous records are retained as Other and exported for review. This is an analytical simplification, not a definitive cultural taxonomy.

## Representations
Visual features use a frozen OpenCLIP ViT-B-32 model. Semantic features use the multilingual MiniLM sentence-transformer over `text_for_embedding`, a structured metadata textualization rather than the museum's original narrative description. Embeddings are L2-normalized. The pilot multimodal baseline concatenates 0.5 × visual and 0.5 × text representations; these weights are configurable and do not imply equal cultural contribution.

## Similarity, clustering and network
Cosine similarity is calculated within each modality and for the concatenated baseline. Pair tables distinguish visual-semantic alignment from mismatch. KMeans is compared for K=2–8 using silhouette scores. The similarity network connects each object to its top five multimodal neighbors; centrality and communities are exploratory computational structures. Nodes with cross-context connectivity are called bridge candidates, not historical intermediaries.

## Temporal analysis and reproducibility
Only museum-provided dates are used. `mid_year` is the mean of valid beginning and ending dates, with implausible values marked missing. Counts are summarized by century. Random processes use seed 42. Model names, runtime information, indices, intermediate tables, figures, and reports are saved under the repository.

