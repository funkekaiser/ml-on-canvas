# ml-on-canvas

**ML algorithms visualized in vanilla canvas — zero dependencies.**

Five classic classifiers cycle through a gallery, each one **trained live in your browser** on a freshly generated 2-D point cloud and rendered to a plain `<canvas>`: k-means++, softmax regression, linear SVM (one-vs-rest), k-NN, and a gini decision tree. No ML library, no framework, no build step — one ES module, ~600 lines.

**[▶ Live demo](https://funkekaiser.github.io/ml-on-canvas/)**

## What you're watching

Every round spawns five fresh gaussian clusters, then trains a real model on them while you watch:

| Algorithm | What actually runs |
|---|---|
| **k-means++** | Centroids seeded ∝ squared distance (the real k-means++ init), then animated Lloyd iterations — assign, move, repeat until centroids stop moving. Dashed trails show each centroid's path. |
| **softmax regression** | Full-batch gradient descent on cross-entropy, one epoch per frame, with the live decision regions and training accuracy in the readout. |
| **linear SVM (OvR)** | One-vs-rest hinge loss with the **Pegasos** learning-rate schedule (`η = 1/(λt)`). Trains past argmax-accuracy until the margin scale converges, then rings the support vectors (margin < 1.05). |
| **k-NN** | The decision boundary morphs as k steps 1 → 3 → 5 → 7 → 9, showing how larger neighborhoods smooth the regions. |
| **decision tree** | CART-style axis-aligned splits chosen by gini impurity (16 candidate thresholds per axis), revealed level by level to depth 3. |

Decision regions are rendered to one offscreen canvas per class and masked with a radial falloff around the data, so the visualization stays a quiet background rather than tinting the whole page. Under `prefers-reduced-motion` the engine renders a single static, fully-converged frame instead of animating.

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
| `algorithms` | all five | Subset/order of `["kmeans","softmax","svm","knn","tree"]`. |
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
