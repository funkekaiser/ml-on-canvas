# ML on canvas

**ML algorithms visualized in vanilla canvas - zero dependencies.**

Seven classic classifiers cycle through a gallery, each one **trained live in your browser** on a freshly generated 2-D point cloud and rendered to a plain `<canvas>`: k-means++, softmax regression, linear SVM (one-vs-rest), k-NN, a tiny MLP, a gini decision tree, and a gaussian mixture fit by EM. No ML library, no framework, no build step.

**[▶ Live demo](https://funkekaiser.github.io/ml-on-canvas/)**

![softmax regression training live on a 2-D point cloud, decision regions converging onto five clusters](assets/demo.gif)

## What you're watching

Every round spawns five fresh clusters, then trains a real model on them while you watch. The linear models always get gaussian blobs; the nonlinear ones (k-NN, MLP, tree) occasionally get **moons, rings, or anisotropic clouds** — data a straight line can't carve, where their decision boundaries show real character.

| Algorithm | What actually runs |
|---|---|
| **k-means++** | Centroids seeded ∝ squared distance (the real k-means++ init), then animated Lloyd iterations — assign, move, repeat until centroids stop moving. Dashed trails show each centroid's path; the readout tracks the inertia (WCSS) falling per iteration. |
| **softmax regression** | Full-batch gradient descent on cross-entropy, one epoch per frame, with the live decision regions and the cross-entropy loss + training accuracy in the readout. |
| **linear SVM (OvR)** | One-vs-rest hinge loss with the **Pegasos** learning-rate schedule (`η = 1/(λt)`). The round ends when the weights stop drifting (relative movement < 1.2% per 60-epoch window) — the margin settles instead of a timer running out — then the support vectors get rings (margin < 1.05). |
| **k-NN** | The decision boundary morphs as k steps 1 → 3 → 5 → 7 → 9, showing how larger neighborhoods smooth the regions. |
| **MLP** | A tiny 2 → 8 → K tanh network trained by full-batch gradient descent on cross-entropy — curved decision boundaries next to all the linear/axis-aligned ones. |
| **decision tree** | CART-style axis-aligned splits chosen by gini impurity (16 candidate thresholds per axis). Each split *sweeps* its candidate threshold line across the node before locking in — you watch the reasoning, not just the result — then the regions split, level by level to depth 3. |
| **GMM (EM)** | A full-covariance gaussian mixture fit by expectation-maximization. Points blend between cluster colors by responsibility (soft assignment), and each component draws animated 1σ/2σ covariance ellipses that stretch to fit the data. |

Decision regions are predicted on a coarse grid into one tiny offscreen canvas per class, scaled up with image smoothing off (same blocky look, ~10× less fill work), and masked with a radial falloff around the data, so the visualization stays a quiet background rather than tinting the whole page. Under `prefers-reduced-motion` the engine renders a single static, fully-converged frame instead of animating. After about three rounds with no pointer activity the gallery pauses on a converged frame — a resting plot, not a screensaver — and resumes on the next interaction.

## Usage

Copy `ml-on-canvas.js` anywhere (or add this repo as a git submodule) and:

```html
<canvas id="viz"></canvas>
<script type="module">
  import { createGallery } from "./ml-on-canvas.js";

  const gallery = createGallery(document.getElementById("viz"), {
    theme: "dark",                              // "light" (default) | "dark"
    onStatus: (label) => console.log(label),    // e.g. "linear svm (ovr) · epoch 0840"
    onHover: (k) => highlightThing(k),          // hovered cluster index or null
  });

  gallery.setAlgorithm("svm");   // jump the gallery
  gallery.setTheme("light");     // re-render regions in the other palette
  gallery.setHover(2);           // highlight a cluster from your own UI
</script>
```

The engine treats the canvas as a fullscreen background: it sizes itself to the window, handles devicePixelRatio, and starts a fresh round on real resizes (mobile URL-bar jitter is ignored).

### Options

All optional — the defaults reproduce the demo.

| Option | Default | |
|---|---|---|
| `theme` | `"light"` | Active palette name. |
| `themes` | `DEFAULT_THEMES` | `{ light: {...}, dark: {...} }` — each palette has `bg`, `ink`, `ring`, `gray`, and `pal` (one `[r,g,b]` per cluster). |
| `algorithms` | all seven | Subset/order of `["kmeans","softmax","svm","knn","mlp","tree","gmm"]`. |
| `clusters` | `5` | K. Palettes must supply K colors. |
| `pointsPerCluster` | `48` | |
| `respectReducedMotion` | `true` | Static converged frame under `prefers-reduced-motion`. |
| `placement(w, h)` | centered | Returns `{ x: [lo, hi], y: [lo, hi], sep }` in unit coordinates — where cluster centers may spawn. Useful to keep the data clear of overlaid text. |
| `tapIgnore` | `"a, button, input, select, label"` | Selector whose taps don't trigger the mobile cluster-flash. |
| `onStatus(label)` | — | Fires when the readout text changes. |
| `onHover(k)` | — | Fires when the hovered cluster changes (`0..K-1` or `null`). Hover comes from the pointer (nearest cluster within 170px), taps on touch devices, or your own `setHover()`. |

### Controller

`createGallery` returns `{ start, stop, destroy, setTheme(name), setHover(k|null), setAlgorithm(name), regenerate(), getAlgorithm() }`. `destroy()` removes all window/document listeners.

## Used in the wild

This engine is the animated background of [funkekaiser.com](https://funkekaiser.com), where it's consumed as a git submodule — each page link owns one cluster, and hovering a link highlights its cluster (and vice versa) via `setHover`/`onHover`.

## License

[MIT](LICENSE)
