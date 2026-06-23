| Model   | Split           | Task Set (OLMES task IDs)                                              | Why These                                   |
| ------- | --------------- | ---------------------------------------------------------------------- | ------------------------------------------- |
| **1B**  | Single          | `arc_easy:rc::olmes`<br>`piqa:rc::olmes`<br>`hellaswag:mc::olmes`      | Core MCQA signal with minimal cost          |
| **3B**  | Split-A         | `arc_easy:rc::olmes`<br>`piqa:rc::olmes`<br>`hellaswag:mc::olmes`      | Baseline improvements                       |
|         | Split-B         | `arc_challenge:rc::olmes`<br>`csqa:rc::olmes`<br>`socialiqa:rc::olmes` | Early reasoning / commonsense QA            |
| **8B**  | Split-A         | `arc_challenge:rc::olmes`<br>`triviaqa::olmes`<br>`naturalqs::olmes`   | Deeper QA + knowledge coverage              |
|         | Split-B         | `csqa:rc::olmes`<br>`socialiqa:rc::olmes`<br>`winogrande:rc::olmes`    | Broader commonsense / pronoun resolution    |
| **70B** | Split-A (mid)   | `arc_challenge:rc::olmes`<br>`triviaqa::olmes`<br>`naturalqs::olmes`   | Regression guard on core tasks              |
|         | Split-B (final) | `gsm8k::olmes`<br>`mbpp::none`<br>`bbh_mmlu_subset`*                   | Hard math / code / reasoning (small slices) |
|         | Split-C (final) | `squad::olmes`<br>`jeopardy::olmes`                                    | Broader QA and general knowledge            |
