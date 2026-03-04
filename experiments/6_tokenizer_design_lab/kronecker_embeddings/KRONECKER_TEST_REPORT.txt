================================================================================
  KRONECKER EMBEDDINGS - COMPREHENSIVE TEST SUITE
  Using: tokenizer.json + gptoss_kronecker_embeddings_bf16.pt
================================================================================

  Loading source files...
  Encoder: CHAR_DIM=256, POS_DIM=32, D=8192
  Cache: torch.Size([131072, 8192]), dtype=torch.bfloat16
  Vocab mapping: 131,072 tokens (356 special)
  Load time: 2.64s
  Testable tokens: 129,359

================================================================================
TEST 1: Encode == Decode Fidelity (Curated Tokens from .json + .pt)
================================================================================
    ID  18547 | 'haus'                    ( 4B) | RT:OK PT:OK
    ID   3484 | 'utes'                    ( 4B) | RT:OK PT:OK
    ID  50837 | 'utte'                    ( 4B) | RT:OK PT:OK
  [PASS] English ASCII: 20 tokens | RT=20/20, PT=20/20

  [PASS] Devanagari: 0 tokens | RT=0/0, PT=0/0

    ID  83643 | '@Setter'                 ( 7B) | RT:OK PT:OK
    ID  46950 | '.Token'                  ( 6B) | RT:OK PT:OK
    ID  89494 | ',.

'                    ( 4B) | RT:OK PT:OK
  [PASS] Code / Operators: 20 tokens | RT=20/20, PT=20/20

    ID   6946 | ' ::'                     ( 3B) | RT:OK PT:OK
    ID 112717 | ':The'                    ( 4B) | RT:OK PT:OK
    ID  69881 | '[w'                      ( 2B) | RT:OK PT:OK
  [PASS] JSON / Brackets: 20 tokens | RT=20/20, PT=20/20

    ID  12669 | ' mea'                    ( 4B) | RT:OK PT:OK
    ID 114865 | ' enthr'                  ( 6B) | RT:OK PT:OK
    ID  33362 | ' compounds'              (10B) | RT:OK PT:OK
  [PASS] Whitespace / Special: 20 tokens | RT=20/20, PT=20/20

    ID  27969 | ' à¬'                     ( 5B) | RT:OK PT:OK
    ID 125959 | ' à´®àµĩ'                 (13B) | RT:OK PT:OK
    ID 126947 | 'àªªà«ģàª°'               (18B) | RT:OK PT:OK
  [PASS] Mixed / Other: 20 tokens | RT=20/20, PT=20/20


  Result: RT 100/100, PT 100/100
  PASS

================================================================================
TEST 2: Batch Encode/Decode Consistency (Real Vocab + .pt)
================================================================================
  [PASS] Batch size=5: RT mismatches=0, PT mismatches=0
    [OK] ID 105524: 'qry'                -> 'qry'
    [OK] ID  41348: 'Answers'            -> 'Answers'
    [OK] ID  52583: 'Ã¤chen'             -> 'Ã¤chen'
  [PASS] Batch size=16: RT mismatches=0, PT mismatches=0
    [OK] ID  27653: ' prostit'           -> ' prostit'
    [OK] ID 119723: ' à¤ľà¤Ĺ'            -> ' à¤ľà¤Ĺ'
    [OK] ID 123922: ' à¤¸à¤¾à¤¹'         -> ' à¤¸à¤¾à¤¹'
  [PASS] Batch size=64: RT mismatches=0, PT mismatches=0
    [OK] ID  34719: ' Jed'               -> ' Jed'
    [OK] ID  18301: ' PER'               -> ' PER'
    [OK] ID  32326: '_rem'               -> '_rem'

  Result: PASS

================================================================================
TEST 3: Speed Matrix (Batch Size x Sequence Length)
================================================================================
  Vocabulary pool: 129359 tokens
  Batch sizes: [1, 4, 16, 64, 256]
  Sequence lengths: [1, 8, 32, 128, 512, 1024, 2048, 4096, 8192]
  Max tokens/cell: 131,072

   Batch  SeqLen   Tokens    Enc(ms)    Dec(ms)      Enc tok/s      Dec tok/s   Fidelity
  ------------------------------------------------------------------------------------------
       1       1        1       0.08       0.11         12,453          8,818     100.0%
       1       8        8       0.15       0.26         54,348         31,068     100.0%
       1      32       32       0.96       0.94         33,441         33,977     100.0%
       1     128      128       3.96       3.73         32,336         34,351     100.0%
       1     512      512      22.88      25.75         22,375         19,882     100.0%
       1    1024     1024      44.33      43.72         23,100         23,420     100.0%
       1    2048     2048      75.35      52.85         27,179         38,751     100.0%
       1    4096     4096     123.78     107.62         33,091         38,061     100.0%
       1    8192     8192     326.84     184.12         25,064         44,492     100.0%
       4       1        4       8.53       0.28            469         14,099     100.0%
       4       8       32       0.55       0.67         57,940         47,833     100.0%
       4      32      128       2.32       3.54         55,244         36,109     100.0%
       4     128      512      17.38      19.29         29,455         26,539     100.0%
       4     512     2048      87.84      53.75         23,315         38,101     100.0%
       4    1024     4096     123.34      92.63         33,208         44,221     100.0%
       4    2048     8192     244.54     184.72         33,500         44,348     100.0%
       4    4096    16384     854.71     379.70         19,169         43,150     100.0%
       4    8192    32768    1442.59     797.75         22,715         41,076     100.0%
      16       1       16      36.99       0.81            433         19,697     100.0%
      16       8      128       2.53       3.87         50,685         33,101     100.0%
      16      32      512      24.46      18.28         20,929         28,013     100.0%
      16     128     2048      83.03      48.95         24,665         41,838     100.0%
      16     512     8192     254.36     217.07         32,206         37,738     100.0%
      16    1024    16384     543.10     375.57         30,168         43,625     100.0%
      16    2048    32768    1439.19     742.77         22,768         44,116     100.0%
      16    4096    65536    5400.24    1501.26         12,136         43,654     100.0%
      16    8192   131072   26363.57    3583.49          4,972         36,577     100.0%
      64       1       64     289.60       3.13            221         20,445     100.0%
      64       8      512      10.91      13.15         46,940         38,934     100.0%
      64      32     2048      45.80      50.30         44,715         40,718     100.0%
      64     128     8192     223.87     180.03         36,592         45,503     100.0%
      64     512    32768    1531.11    2462.57         21,401         13,306     100.0%
      64    1024    65536   16382.47    7387.62          4,000          8,871     100.0%
      64    2048   131072   68247.11   14847.01          1,921          8,828     100.0%
      64    4096   262144   --SKIP--  (OOM cap)
      64    8192   524288   --SKIP--  (OOM cap)
     256       1      256     541.96      49.85            472          5,136     100.0%
     256       8     2048     135.50     246.49         15,115          8,309     100.0%
     256      32     8192     509.71     856.80         16,072          9,561     100.0%
     256     128    32768    4407.34    3750.19          7,435          8,738     100.0%
     256     512   131072   62671.04   14443.98          2,091          9,075     100.0%
     256    1024   262144   --SKIP--  (OOM cap)
     256    2048   524288   --SKIP--  (OOM cap)
     256    4096  1048576   --SKIP--  (OOM cap)
     256    8192  2097152   --SKIP--  (OOM cap)

  Result: PASS

================================================================================
TEST 4: Pre-computed .pt Cache vs Real-time Encoding
================================================================================
  Cache shape: torch.Size([131072, 8192]), dtype: torch.bfloat16
  Size: 2.15 GB

  Embedding match (RT vs .pt): 200/200 (100.0%)
  Decode from .pt fidelity:    200/200 (100.0%)

  Cache Lookup Speed:
     Batch   Lookup(ms)     Tokens/sec
  ----------------------------------------
         1       0.0963         10,389
        16       0.1300        123,061
        64       0.2420        264,472
       256       0.6069        421,810
      1024       2.4214        422,895
      4096       9.9996        409,615

  Result: PASS

================================================================================
TEST 5: Full Vocabulary Round-trip (ALL tokens, both files)
================================================================================
  Total mapped tokens: 131,072
  Special tokens: 356
  Testable tokens: 129,359

  Time: 38.84s (3,331 tokens/sec)
  Real-time: 129,359/129,359 passed (100.00%)
  From .pt:  129,359/129,359 passed (100.00%)

  Example successful round-trips:
    ID      0 | '!'  (1B) -> encode -> decode -> '!'  [RT+PT OK]
    ID      1 | '"'  (1B) -> encode -> decode -> '"'  [RT+PT OK]
    ID      2 | '#'  (1B) -> encode -> decode -> '#'  [RT+PT OK]
    ID      3 | '$'  (1B) -> encode -> decode -> '$'  [RT+PT OK]
    ID      4 | '%'  (1B) -> encode -> decode -> '%'  [RT+PT OK]

  Result: PASS

================================================================================
TEST 6: Mathematical Properties (using .pt embeddings)
================================================================================
  [PASS] Dimension: (8192,) (expected (8192,))
  [PASS] Unit norm: .pt embeddings have ||emb|| ~ 1.0
    dot('_PI', '.PARAM') = 0.235992
    dot('ilai', ' fogo') = 0.000000
    dot(' fuga', ' à°ķà±ģà°®') = 0.102644
    dot(' eye', ' tors') = 0.223633
    dot(' Nathan', ' ellen') = 0.154671
  [PASS] Near-orthogonality: dot products < 0.5
  [PASS] RT encode matches .pt cache (bf16 tolerance)
  [PASS] Empty string -> zero vector (norm=0.00e+00)

  Result: PASS

================================================================================
TEST 7: End-to-End .pt -> decode_word -> tokenizer.json (ALL tokens)
================================================================================
  Testable tokens: 129,359

  Time: 23.21s (5,573 tokens/sec)
  Passed: 129,359/129,359 (100.00%)
  Failed: 0/129,359

  Result: PASS

================================================================================
TEST 8: Full Training Pipeline (Raw Text -> Tokens -> Embeddings -> Decode)
================================================================================
  Tokenizer loaded: vocab_size=131,072
  Embeddings: torch.Size([131072, 8192]), dtype=torch.bfloat16

  --- English ---
  Input: 'The quick brown fox jumps over the lazy dog. Machine learning models process tex'...
  Step 1 - Tokenize: 17 tokens in 6.479ms
    [ 0] ID=   864  text='The'
    [ 1] ID=  4000  text=' quick'
    [ 2] ID= 15431  text=' brown'
    [ 3] ID= 51446  text=' fox'
    [ 4] ID= 49516  text=' jumps'
    [ 5] ID=   945  text=' over'
    [ 6] ID=   290  text=' the'
    [ 7] ID= 22514  text=' lazy'
    [ 8] ID=  5234  text=' dog'
    [ 9] ID=    13  text='.'
    [10] ID= 14981  text=' Machine'
    [11] ID=  6072  text=' learning'
    [12] ID=  5691  text=' models'
    [13] ID=  1935  text=' process'
    [14] ID=  1873  text=' text'
    [15] ID= 15875  text=' tokens'
    [16] ID=    13  text='.'
  Step 2 - .pt Lookup: shape=(17, 8192), dtype=torch.bfloat16, time=0.616ms
  Metrics:
    Avg norm:     1.0005 (expected ~1.0)
    Norm range:   [0.9999, 1.0025]
    Sparsity:     99.94% zeros
    Memory:       544.0 KB (bf16: 272.0 KB)
  Step 3 - Decode ALL from .pt: 17/17 match (100.0%) [OK]
    [OK] ID    864: .pt -> decode -> 'The'  [match: True]
    [OK] ID   4000: .pt -> decode -> ' quick'  [match: True]
    [OK] ID  15431: .pt -> decode -> ' brown'  [match: True]
    [OK] ID  51446: .pt -> decode -> ' fox'  [match: True]
    [OK] ID  49516: .pt -> decode -> ' jumps'  [match: True]
    [OK] ID    945: .pt -> decode -> ' over'  [match: True]
    [OK] ID    290: .pt -> decode -> ' the'  [match: True]
    [OK] ID  22514: .pt -> decode -> ' lazy'  [match: True]
    [OK] ID   5234: .pt -> decode -> ' dog'  [match: True]
    [OK] ID     13: .pt -> decode -> '.'  [match: True]
    [OK] ID  14981: .pt -> decode -> ' Machine'  [match: True]
    [OK] ID   6072: .pt -> decode -> ' learning'  [match: True]
    [OK] ID   5691: .pt -> decode -> ' models'  [match: True]
    [OK] ID   1935: .pt -> decode -> ' process'  [match: True]
    [OK] ID   1873: .pt -> decode -> ' text'  [match: True]
    [OK] ID  15875: .pt -> decode -> ' tokens'  [match: True]
    [OK] ID     13: .pt -> decode -> '.'  [match: True]

  --- Code (Python) ---
  Input: 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fi'...
  Step 1 - Tokenize: 31 tokens in 1.503ms
    [ 0] ID=  1145  text='def'
    [ 1] ID= 14731  text=' fib'
    [ 2] ID=114071  text='onacci'
    [ 3] ID=  2044  text='(n'
    [ 4] ID=  1607  text='):
'
    [ 5] ID=   271  text='   '
    [ 6] ID=   512  text=' if'
    [ 7] ID=   297  text=' n'
    [ 8] ID=  4121  text=' <='
    [ 9] ID=   220  text=' '
    [10] ID=    16  text='1'
    [11] ID=   674  text=':
'
    [12] ID=   308  text='       '
    [13] ID=   587  text=' return'
    [14] ID=   297  text=' n'
    [15] ID=   198  text='
'
    [16] ID=   271  text='   '
    [17] ID=   587  text=' return'
    [18] ID= 14731  text=' fib'
    [19] ID=114071  text='onacci'
    [20] ID=  2044  text='(n'
    [21] ID=    12  text='-'
    [22] ID=    16  text='1'
    [23] ID=     8  text=')'
    [24] ID=   617  text=' +'
    [25] ID= 14731  text=' fib'
    [26] ID=114071  text='onacci'
    [27] ID=  2044  text='(n'
    [28] ID=    12  text='-'
    [29] ID=    17  text='2'
    [30] ID=     8  text=')'
  Step 2 - .pt Lookup: shape=(31, 8192), dtype=torch.bfloat16, time=0.414ms
  Metrics:
    Avg norm:     1.0005 (expected ~1.0)
    Norm range:   [0.9999, 1.0025]
    Sparsity:     99.96% zeros
    Memory:       992.0 KB (bf16: 496.0 KB)
  Step 3 - Decode ALL from .pt: 31/31 match (100.0%) [OK]
    [OK] ID   1145: .pt -> decode -> 'def'  [match: True]
    [OK] ID  14731: .pt -> decode -> ' fib'  [match: True]
    [OK] ID 114071: .pt -> decode -> 'onacci'  [match: True]
    [OK] ID   2044: .pt -> decode -> '(n'  [match: True]
    [OK] ID   1607: .pt -> decode -> '):
'  [match: True]
    [OK] ID    271: .pt -> decode -> '   '  [match: True]
    [OK] ID    512: .pt -> decode -> ' if'  [match: True]
    [OK] ID    297: .pt -> decode -> ' n'  [match: True]
    [OK] ID   4121: .pt -> decode -> ' <='  [match: True]
    [OK] ID    220: .pt -> decode -> ' '  [match: True]
    [OK] ID     16: .pt -> decode -> '1'  [match: True]
    [OK] ID    674: .pt -> decode -> ':
'  [match: True]
    [OK] ID    308: .pt -> decode -> '       '  [match: True]
    [OK] ID    587: .pt -> decode -> ' return'  [match: True]
    [OK] ID    297: .pt -> decode -> ' n'  [match: True]
    [OK] ID    198: .pt -> decode -> '
'  [match: True]
    [OK] ID    271: .pt -> decode -> '   '  [match: True]
    [OK] ID    587: .pt -> decode -> ' return'  [match: True]
    [OK] ID  14731: .pt -> decode -> ' fib'  [match: True]
    [OK] ID 114071: .pt -> decode -> 'onacci'  [match: True]
    [OK] ID   2044: .pt -> decode -> '(n'  [match: True]
    [OK] ID     12: .pt -> decode -> '-'  [match: True]
    [OK] ID     16: .pt -> decode -> '1'  [match: True]
    [OK] ID      8: .pt -> decode -> ')'  [match: True]
    [OK] ID    617: .pt -> decode -> ' +'  [match: True]
    [OK] ID  14731: .pt -> decode -> ' fib'  [match: True]
    [OK] ID 114071: .pt -> decode -> 'onacci'  [match: True]
    [OK] ID   2044: .pt -> decode -> '(n'  [match: True]
    [OK] ID     12: .pt -> decode -> '-'  [match: True]
    [OK] ID     17: .pt -> decode -> '2'  [match: True]
    [OK] ID      8: .pt -> decode -> ')'  [match: True]

  --- Code (JSON) ---
  Input: '{"model": "kronecker", "vocab_size": 131072, "dim": 8192, "layers": 80}'
  Step 1 - Tokenize: 30 tokens in 0.160ms
    [ 0] ID=  8666  text='{"'
    [ 1] ID=  4170  text='model'
    [ 2] ID=  1088  text='":'
    [ 3] ID=   381  text=' "'
    [ 4] ID=  5914  text='kr'
    [ 5] ID=   640  text='one'
    [ 6] ID= 19342  text='cker'
    [ 7] ID=   626  text='",'
    [ 8] ID=   381  text=' "'
    [ 9] ID=    85  text='v'
    [10] ID= 43315  text='ocab'
    [11] ID=  4143  text='_size'
    [12] ID=  1088  text='":'
    [13] ID=   220  text=' '
    [14] ID= 12903  text='131'
    [15] ID= 32567  text='072'
    [16] ID=    11  text=','
    [17] ID=   381  text=' "'
    [18] ID= 17322  text='dim'
    [19] ID=  1088  text='":'
    [20] ID=   220  text=' '
    [21] ID= 29054  text='819'
    [22] ID=    17  text='2'
    [23] ID=    11  text=','
    [24] ID=   381  text=' "'
    [25] ID= 73965  text='layers'
    [26] ID=  1088  text='":'
    [27] ID=   220  text=' '
    [28] ID=  1908  text='80'
    [29] ID=    92  text='}'
  Step 2 - .pt Lookup: shape=(30, 8192), dtype=torch.bfloat16, time=0.175ms
  Metrics:
    Avg norm:     1.0002 (expected ~1.0)
    Norm range:   [0.9999, 1.0013]
    Sparsity:     99.97% zeros
    Memory:       960.0 KB (bf16: 480.0 KB)
  Step 3 - Decode ALL from .pt: 30/30 match (100.0%) [OK]
    [OK] ID   8666: .pt -> decode -> '{"'  [match: True]
    [OK] ID   4170: .pt -> decode -> 'model'  [match: True]
    [OK] ID   1088: .pt -> decode -> '":'  [match: True]
    [OK] ID    381: .pt -> decode -> ' "'  [match: True]
    [OK] ID   5914: .pt -> decode -> 'kr'  [match: True]
    [OK] ID    640: .pt -> decode -> 'one'  [match: True]
    [OK] ID  19342: .pt -> decode -> 'cker'  [match: True]
    [OK] ID    626: .pt -> decode -> '",'  [match: True]
    [OK] ID    381: .pt -> decode -> ' "'  [match: True]
    [OK] ID     85: .pt -> decode -> 'v'  [match: True]
    [OK] ID  43315: .pt -> decode -> 'ocab'  [match: True]
    [OK] ID   4143: .pt -> decode -> '_size'  [match: True]
    [OK] ID   1088: .pt -> decode -> '":'  [match: True]
    [OK] ID    220: .pt -> decode -> ' '  [match: True]
    [OK] ID  12903: .pt -> decode -> '131'  [match: True]
    [OK] ID  32567: .pt -> decode -> '072'  [match: True]
    [OK] ID     11: .pt -> decode -> ','  [match: True]
    [OK] ID    381: .pt -> decode -> ' "'  [match: True]
    [OK] ID  17322: .pt -> decode -> 'dim'  [match: True]
    [OK] ID   1088: .pt -> decode -> '":'  [match: True]
    [OK] ID    220: .pt -> decode -> ' '  [match: True]
    [OK] ID  29054: .pt -> decode -> '819'  [match: True]
    [OK] ID     17: .pt -> decode -> '2'  [match: True]
    [OK] ID     11: .pt -> decode -> ','  [match: True]
    [OK] ID    381: .pt -> decode -> ' "'  [match: True]
    [OK] ID  73965: .pt -> decode -> 'layers'  [match: True]
    [OK] ID   1088: .pt -> decode -> '":'  [match: True]
    [OK] ID    220: .pt -> decode -> ' '  [match: True]
    [OK] ID   1908: .pt -> decode -> '80'  [match: True]
    [OK] ID     92: .pt -> decode -> '}'  [match: True]

  --- Devanagari (Hindi) ---
  Input: 'भारत एक विविधताओं से भरा देश है। यहाँ अनेक भाषाएँ बोली जाती हैं।'
  Step 1 - Tokenize: 19 tokens in 0.321ms
    [ 0] ID=125065  text='भारत'
    [ 1] ID=117389  text=' एक'
    [ 2] ID=127781  text=' विविध'
    [ 3] ID=127243  text='ताओं'
    [ 4] ID=117259  text=' से'
    [ 5] ID=117213  text=' भ'
    [ 6] ID=118219  text='रा'
    [ 7] ID=118528  text=' देश'
    [ 8] ID=117159  text=' है'
    [ 9] ID=117127  text='।'
    [10] ID=122266  text=' यहाँ'
    [11] ID=123264  text=' अनेक'
    [12] ID=121260  text=' भाष'
    [13] ID=127347  text='ाएँ'
    [14] ID=119960  text=' बो'
    [15] ID=117780  text='ली'
    [16] ID=120521  text=' जाती'
    [17] ID=117398  text=' हैं'
    [18] ID=117127  text='।'
  Step 2 - .pt Lookup: shape=(19, 8192), dtype=torch.bfloat16, time=0.123ms
  Metrics:
    Avg norm:     1.0004 (expected ~1.0)
    Norm range:   [0.9985, 1.0025]
    Sparsity:     99.79% zeros
    Memory:       608.0 KB (bf16: 304.0 KB)
  Step 3 - Decode ALL from .pt: 19/19 match (100.0%) [OK]
    [OK] ID 125065: .pt -> decode -> 'भारत'  [match: True]
    [OK] ID 117389: .pt -> decode -> ' एक'  [match: True]
    [OK] ID 127781: .pt -> decode -> ' विविध'  [match: True]
    [OK] ID 127243: .pt -> decode -> 'ताओं'  [match: True]
    [OK] ID 117259: .pt -> decode -> ' से'  [match: True]
    [OK] ID 117213: .pt -> decode -> ' भ'  [match: True]
    [OK] ID 118219: .pt -> decode -> 'रा'  [match: True]
    [OK] ID 118528: .pt -> decode -> ' देश'  [match: True]
    [OK] ID 117159: .pt -> decode -> ' है'  [match: True]
    [OK] ID 117127: .pt -> decode -> '।'  [match: True]
    [OK] ID 122266: .pt -> decode -> ' यहाँ'  [match: True]
    [OK] ID 123264: .pt -> decode -> ' अनेक'  [match: True]
    [OK] ID 121260: .pt -> decode -> ' भाष'  [match: True]
    [OK] ID 127347: .pt -> decode -> 'ाएँ'  [match: True]
    [OK] ID 119960: .pt -> decode -> ' बो'  [match: True]
    [OK] ID 117780: .pt -> decode -> 'ली'  [match: True]
    [OK] ID 120521: .pt -> decode -> ' जाती'  [match: True]
    [OK] ID 117398: .pt -> decode -> ' हैं'  [match: True]
    [OK] ID 117127: .pt -> decode -> '।'  [match: True]

  --- Mixed (English + Hindi) ---
  Input: 'The word नमस्ते means hello in Hindi. भारत is called India in English.'
  Step 1 - Tokenize: 17 tokens in 0.574ms
    [ 0] ID=   864  text='The'
    [ 1] ID=  1868  text=' word'
    [ 2] ID=123431  text=' नम'
    [ 3] ID=117866  text='स्त'
    [ 4] ID=117075  text='े'
    [ 5] ID=  3920  text=' means'
    [ 6] ID= 31133  text=' hello'
    [ 7] ID=   305  text=' in'
    [ 8] ID= 29338  text=' Hindi'
    [ 9] ID=    13  text='.'
    [10] ID=118678  text=' भारत'
    [11] ID=   372  text=' is'
    [12] ID=  3613  text=' called'
    [13] ID=  6749  text=' India'
    [14] ID=   305  text=' in'
    [15] ID=  6226  text=' English'
    [16] ID=    13  text='.'
  Step 2 - .pt Lookup: shape=(17, 8192), dtype=torch.bfloat16, time=0.135ms
  Metrics:
    Avg norm:     1.0004 (expected ~1.0)
    Norm range:   [0.9985, 1.0025]
    Sparsity:     99.91% zeros
    Memory:       544.0 KB (bf16: 272.0 KB)
  Step 3 - Decode ALL from .pt: 17/17 match (100.0%) [OK]
    [OK] ID    864: .pt -> decode -> 'The'  [match: True]
    [OK] ID   1868: .pt -> decode -> ' word'  [match: True]
    [OK] ID 123431: .pt -> decode -> ' नम'  [match: True]
    [OK] ID 117866: .pt -> decode -> 'स्त'  [match: True]
    [OK] ID 117075: .pt -> decode -> 'े'  [match: True]
    [OK] ID   3920: .pt -> decode -> ' means'  [match: True]
    [OK] ID  31133: .pt -> decode -> ' hello'  [match: True]
    [OK] ID    305: .pt -> decode -> ' in'  [match: True]
    [OK] ID  29338: .pt -> decode -> ' Hindi'  [match: True]
    [OK] ID     13: .pt -> decode -> '.'  [match: True]
    [OK] ID 118678: .pt -> decode -> ' भारत'  [match: True]
    [OK] ID    372: .pt -> decode -> ' is'  [match: True]
    [OK] ID   3613: .pt -> decode -> ' called'  [match: True]
    [OK] ID   6749: .pt -> decode -> ' India'  [match: True]
    [OK] ID    305: .pt -> decode -> ' in'  [match: True]
    [OK] ID   6226: .pt -> decode -> ' English'  [match: True]
    [OK] ID     13: .pt -> decode -> '.'  [match: True]

  --- Code + Comments ---
  Input: "# Calculate sum\ndef add(a, b):\n    '''Add two numbers'''\n    return a + b  # sim"...
  Step 1 - Tokenize: 25 tokens in 0.159ms
    [ 0] ID=     2  text='#'
    [ 1] ID= 29875  text=' Calculate'
    [ 2] ID=  3498  text=' sum'
    [ 3] ID=   198  text='
'
    [ 4] ID=  1145  text='def'
    [ 5] ID=  1006  text=' add'
    [ 6] ID=  5111  text='(a'
    [ 7] ID=    11  text=','
    [ 8] ID=   287  text=' b'
    [ 9] ID=  1607  text='):
'
    [10] ID=   271  text='   '
    [11] ID= 17478  text=' ''''
    [12] ID=  2189  text='Add'
    [13] ID=  1634  text=' two'
    [14] ID=  6955  text=' numbers'
    [15] ID= 25818  text=''''
'
    [16] ID=   271  text='   '
    [17] ID=   587  text=' return'
    [18] ID=   261  text=' a'
    [19] ID=   617  text=' +'
    [20] ID=   287  text=' b'
    [21] ID=   220  text=' '
    [22] ID=   942  text=' #'
    [23] ID=  3883  text=' simple'
    [24] ID=  4785  text=' addition'
  Step 2 - .pt Lookup: shape=(25, 8192), dtype=torch.bfloat16, time=0.141ms
  Metrics:
    Avg norm:     1.0005 (expected ~1.0)
    Norm range:   [0.9999, 1.0025]
    Sparsity:     99.96% zeros
    Memory:       800.0 KB (bf16: 400.0 KB)
  Step 3 - Decode ALL from .pt: 25/25 match (100.0%) [OK]
    [OK] ID      2: .pt -> decode -> '#'  [match: True]
    [OK] ID  29875: .pt -> decode -> ' Calculate'  [match: True]
    [OK] ID   3498: .pt -> decode -> ' sum'  [match: True]
    [OK] ID    198: .pt -> decode -> '
'  [match: True]
    [OK] ID   1145: .pt -> decode -> 'def'  [match: True]
    [OK] ID   1006: .pt -> decode -> ' add'  [match: True]
    [OK] ID   5111: .pt -> decode -> '(a'  [match: True]
    [OK] ID     11: .pt -> decode -> ','  [match: True]
    [OK] ID    287: .pt -> decode -> ' b'  [match: True]
    [OK] ID   1607: .pt -> decode -> '):
'  [match: True]
    [OK] ID    271: .pt -> decode -> '   '  [match: True]
    [OK] ID  17478: .pt -> decode -> ' ''''  [match: True]
    [OK] ID   2189: .pt -> decode -> 'Add'  [match: True]
    [OK] ID   1634: .pt -> decode -> ' two'  [match: True]
    [OK] ID   6955: .pt -> decode -> ' numbers'  [match: True]
    [OK] ID  25818: .pt -> decode -> ''''
'  [match: True]
    [OK] ID    271: .pt -> decode -> '   '  [match: True]
    [OK] ID    587: .pt -> decode -> ' return'  [match: True]
    [OK] ID    261: .pt -> decode -> ' a'  [match: True]
    [OK] ID    617: .pt -> decode -> ' +'  [match: True]
    [OK] ID    287: .pt -> decode -> ' b'  [match: True]
    [OK] ID    220: .pt -> decode -> ' '  [match: True]
    [OK] ID    942: .pt -> decode -> ' #'  [match: True]
    [OK] ID   3883: .pt -> decode -> ' simple'  [match: True]
    [OK] ID   4785: .pt -> decode -> ' addition'  [match: True]

  Overall: 139/139 tokens decoded correctly (100.0%)

  Result: PASS


  [INFO] Report generated: D:\LLM\experiments\6_tokenizer_design_lab\kronecker_embeddings\KRONECKER_TEST_REPORT.md
================================================================================
  FINAL SUMMARY
================================================================================

  [PASS] encode_decode_fidelity
  [PASS] batch_encode_decode
  [PASS] speed_matrix
  [PASS] cache_vs_realtime
  [PASS] full_vocab_roundtrip
  [PASS] mathematical_properties
  [PASS] pt_to_json_e2e
  [PASS] full_pipeline

  *** ALL TESTS PASSED - 100% Encode == Decode ***