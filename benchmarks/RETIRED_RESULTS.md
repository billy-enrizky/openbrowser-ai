# Retired Benchmark Results

## Contents

- [1. Recovery](#1-recovery)
- [2. Current evidence retained](#2-current-evidence-retained)
- [3. Retired result files](#3-retired-result-files)
- [4. Retired charts and scripts](#4-retired-charts-and-scripts)

## 1. Recovery

Cleanup base commit: `ff6d0d8e33418ea9103530806c1e709dccec237f`.

Every file below existed at that commit and was removed from the working tree in the stale-results cleanup. Recover any file with:

```sh
git show ff6d0d8e33418ea9103530806c1e709dccec237f:<path> > <path>
```

The table records the exact size, SHA256, and Git blob ID at the base commit. No force rewrite or history change was applied. The immutable GitHub recovery URL for a path is `https://github.com/billy-enrizky/openbrowser-ai/blob/ff6d0d8e33418ea9103530806c1e709dccec237f/<path>`.

## 2. Current evidence retained

- Interactive cold and warm final4 N=5: `benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final4.*` and `results-warm-five-arm-lean-2026-11-25-n5-final4.*`
- Interactive resource final N=5: `benchmarks/interactive/results/results-interactive-resources-2026-11-25-n5-final.*`
- Firecrawl Local final6: `benchmarks/scraping/results/results-firecrawl-final6-2026-11-25-n10-interleaved.*`
- Lightpanda final6: `benchmarks/scraping/results/results-lightpanda-final6-2026-11-25-n10-interleaved.*`
- Firecrawl Local batch: `benchmarks/scraping/results/results-batch-firecrawl-final-2026-11-24-n10.json`
- No-browser MCP memory: `benchmarks/scraping/results/results-mcp-scrape-web-resource-2026-11-25-n5.json`
- Navigation fixtures: `benchmarks/navigation_results_2026-11-15*.json` and `benchmarks/navigation_pins.example.json`

The newer `ff6d0d8e` native refresh candidate is not committed here. It has three retained CLI cold tail losses, one browser-use warm adapter failure, and one fatal chrome-devtools capture failure. It waits for the warm harness fix and will be published with those losses. Stable-release results above are not replaced by candidate claims.

## 3. Retired result files

| Path | Bytes | SHA256 | Git blob |
| --- | ---: | --- | --- |
| `benchmarks/e2e_4way_cli_results.json` | 88632 | `1b0aa179986d258d4dd1d84c2188fcb7963caa1c2fda9bdab1cf7656986d2e80` | `1d16db97bc27e0f9d9689f622b216054829663a7` |
| `benchmarks/e2e_llm_stats_results.json` | 88630 | `0b5b4c7403d6081335f58e466817cd8f0b5e2b18db1a242555e4a6c67d652331` | `4f1b0282d7f670455c314e0a8c487515947d954d` |
| `benchmarks/interactive/results/results-2026-11-23-extensions.jsonl` | 10567 | `91d1b63877c5bec85bfcd86962dc59e94de9fd8d0e9857b8d0361e863fe50af2` | `e7771eb76650b32301cd64906f7d3ad1bf9661e5` |
| `benchmarks/interactive/results/results-2026-11-23-extensions.summary.json` | 1296 | `6bddf2cc2c12d84600fb7eede0fce03b171fefaf4b32be094112104d722b4132` | `cbebf471def483438e64895cc49f06123232da44` |
| `benchmarks/interactive/results/results-2026-11-23-lean.jsonl` | 39133 | `52965537712fb08cbe51dcb121d1273ae384fdd3401da169e9c871d8fce209a9` | `aff3d8b13bc1b8f26725005d42eedfa7bbfed977` |
| `benchmarks/interactive/results/results-2026-11-23-lean.summary.json` | 5276 | `039deb84874fd3e38fdb335c68e869e92bcf3034aaa69cd483beebb495959aee` | `f9dfdd27e492057b6ce518eda3460475a62c2f19` |
| `benchmarks/interactive/results/results-2026-11-23.jsonl` | 37192 | `211af4ab8f7c106fd1187f70ed273ea8c36ca9a849069eb27e0d75426ef049ca` | `3b04f550312223e3c4719ee5bd638f8aa8cb095a` |
| `benchmarks/interactive/results/results-2026-11-23.summary.json` | 5273 | `3e7abc158f62ee8f741cec8993c7cf0dfec5fb12f2529a48e5d6698456abf47c` | `5c58bc38eaa879fa7097a870e337f84dc2a92231` |
| `benchmarks/interactive/results/results-cli-startup-ab-2026-11-24-n12.jsonl` | 5295 | `7e4a3ca45751b31e942dde40f7400b4e001a2817c8ea1a5c69738aa778173ec5` | `2cc9b3d97b0ba773a01976249e84cc60b3e6c23e` |
| `benchmarks/interactive/results/results-cli-startup-ab-2026-11-24-n12.summary.json` | 856 | `d3334d8f705cd6c367a2b75f29b3b29c5abab264ef78a2bf1e5981a17ec163ef` | `d2b58a225011c9a83c072b76e7c0406c65f0c63d` |
| `benchmarks/interactive/results/results-cold-2026-11-24-n5.jsonl` | 14687 | `5eca60f6c3f42e4317527c60878b939df297bb97437142dab71ee38943985050` | `46406bea79fbcdbe31fcc3a107dd3bc5e0130d6e` |
| `benchmarks/interactive/results/results-cold-2026-11-24-n5.summary.json` | 1039 | `f7b0bf2972886c70d153152e7d14492720e2060e77df3bc750ff1c2ccb26ac92` | `7e20977afa2a3674cdfca77cf10ddd4088bae95c` |
| `benchmarks/interactive/results/results-cold-current-five-arm-2026-11-24-n3.jsonl` | 50857 | `a9dd5d8a85bff2a6406ee18b3e7ea4d03d5feefbabd704a5899256ead8de8e12` | `9801c57ca74dc6239323dca164b28bdbecb1ad8e` |
| `benchmarks/interactive/results/results-cold-current-five-arm-2026-11-24-n3.summary.json` | 6602 | `6c2fff2082ac34c4169a2885ef4dd4c9a6403e34b0a90dded9b1c7292eb6430b` | `578b6dbdd9753b6e787592986e4778a5c32649e4` |
| `benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final2.jsonl` | 108746 | `78237ffaef993d852324cade6f97b97c11e0bf1154c9378b2097ae794986144e` | `a82f0b5b32f0be677cde9112917d4bac81d0f6dd` |
| `benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final2.meta.json` | 514 | `d7feddfd6b4f17f023a1d8e55fd42c442c2fd6aa642763d587a122d348ebcc7d` | `6bb62c9ddab973e7a4d22367cb61884004581c61` |
| `benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final2.summary.json` | 6650 | `ff64e78fa128afa4f97655d4f0df24dcd5b427ea4ded35bd48b54e816500955b` | `0b6f0a08578a51cfe5eb466be6072cbfb20c3218` |
| `benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final3.jsonl` | 108716 | `a6fab83f4679d42de2c2c646e59b97e5177883b3f9db39c0fbfbd3bc02a3d283` | `e966930831cdf7379f13a68dc54912b8c108c7fc` |
| `benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final3.meta.json` | 515 | `2351256935407795c194024973eee78866a612d02f27b31b91e0948fa05ea354` | `c022131e2e14f69e466c0f9f49937c63bdf8c4c6` |
| `benchmarks/interactive/results/results-cold-five-arm-lean-2026-11-25-n5-final3.summary.json` | 6641 | `267175449fa817d82dc5e221d2dc0bb3beb9750894dc1f0dd6cc7cf7f6cfe736` | `40a1f3a75d1749208f790d1524f6489a836bc7f2` |
| `benchmarks/interactive/results/results-cold-six-arm-final-2026-11-24-n3.jsonl` | 52138 | `0633392d4690b22f2f62cb6a80f8d528fb6a5b5479925329c665cac6edfc1594` | `456b9d5d3e32ba1c34ba277525cfb44441e0a8ab` |
| `benchmarks/interactive/results/results-cold-six-arm-final-2026-11-24-n3.summary.json` | 6647 | `34943d9680bd2250fdf235eecf3e87b4a3fcadb19337e5298b5099567992c571` | `55fc95ed18542b9ec2a2bdebdf856a956980e263` |
| `benchmarks/interactive/results/results-cold-six-arm-final2-2026-11-24-n3.jsonl` | 52154 | `4a08cf638bf9dc830928e6880be816b6dec75f66d14facbe014e7a43cb58a81d` | `16187460a0b0f994531154c31dd1b12d3d8b8eaa` |
| `benchmarks/interactive/results/results-cold-six-arm-final2-2026-11-24-n3.summary.json` | 6655 | `2da826aee496162c431b4d44554a0a7c1a0a658925f4a11c717e5e66c0664b69` | `258d5e9195602f0571ff4b219dcee75b5b56e935` |
| `benchmarks/interactive/results/results-cold-six-arm-final3-2026-11-24-n3.jsonl` | 52165 | `b41fb7579f58f663b154de77d500215aa780d7545902d916f39799cc3e683cb3` | `c8d5492ef4857ce764fe17283c217a5680fd80b8` |
| `benchmarks/interactive/results/results-cold-six-arm-final3-2026-11-24-n3.summary.json` | 6652 | `5a0b5a38585b6d4b24f8a5fc3f8a8681249f27467db457335ed3c191428329d2` | `4095b606bf99ce23a679ce0690c82b779671e7f4` |
| `benchmarks/interactive/results/results-cold-six-arm-mcp-2026-11-24-n3.jsonl` | 52142 | `07e9e2a32b50a96fb1204742804409059234a8e950323adb2ae1dad7deda3abc` | `07adccb5a46137c4b5b30a0f04b258d6b7cc6130` |
| `benchmarks/interactive/results/results-cold-six-arm-mcp-2026-11-24-n3.summary.json` | 6652 | `69a8a1cc8b3742b3d67c084b343faa3d786ac87fba202c20f69b0052d1bb53d0` | `61c42771877359dc7ec385f3fa8a164ebbf261fd` |
| `benchmarks/interactive/results/results-cold-six-arm-quiet-2026-11-24-n5.jsonl` | 86912 | `74f625714303b924407c906a50f4c6e43be851dd79b707f794609e6d91691224` | `a087f85a382c268f3462b58e8388ca266c1d4ced` |
| `benchmarks/interactive/results/results-cold-six-arm-quiet-2026-11-24-n5.summary.json` | 6655 | `7be1bf1ca9e174f63293b6b3efbfa6d86e96c497e4a94fd4855677af29fccd40` | `2b843bf70c1e518e276e792f2d284ff171d98f78` |
| `benchmarks/interactive/results/results-interactive-resources-2026-11-25-n5.jsonl` | 726644 | `14ddf50d15d0b282e204f99546fa9f81029b88d6d3d4c36c8a7fa82c6763fbd6` | `c3ef5f6d6413d3614a59be3c30232e89549e68d8` |
| `benchmarks/interactive/results/results-interactive-resources-2026-11-25-n5.summary.json` | 24069 | `64d4a4e38e87f2f0a963eee5024c7f8e9be49fd4874d2976f97b19f9a20f9a8e` | `ff9023cc2dc388977d18b10274a298fb4e0576ca` |
| `benchmarks/interactive/results/results-interactive-resources-pilot-2026-11-25-n1.jsonl` | 150564 | `16f917c7a6a94b9544374e7791c5c2038c92903c55b7cf6663c0df7fc99d94c8` | `4b9722febfb25fc3b2cc0a5efd56a865b173bfed` |
| `benchmarks/interactive/results/results-interactive-resources-pilot-2026-11-25-n1.summary.json` | 24121 | `b100cb812aaac228f1c5ead22e7e03188ed2976898f98d90c3c984cdc21b2706` | `a8ef1ca926c3080e2219b6fdbe29e096aa8bdbaf` |
| `benchmarks/interactive/results/results-paired-download-2026-11-24-n10.jsonl` | 16887 | `e7b1caccbd13f065df16065d3b2a86f05f360cce8d94848f9829509aef768f92` | `5f066708b8afea71b97d8a587d4fd5fd5b008f7e` |
| `benchmarks/interactive/results/results-paired-download-2026-11-24-n10.summary.json` | 1932 | `b126eb3011c8f5b77a76fda8d4c36bc4ad874822c9bb6f1b2af88d12c85c74e7` | `9433c4161f6ad2665e7ac2ca8a9e933f34a4b52d` |
| `benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-2026-11-24-n10.json` | 17002 | `6a55929a16c1b4c2ade3551f4866161258b5862ac6a91d22dc5bf64c4f6dd950` | `6fca79cb3eda7ca24a8c279d6590b9b876ad55c9` |
| `benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-2026-11-24-n10.summary.json` | 1955 | `7a87ab18b8fcc3e34136bdd1f31494699ae5c6c3f2a00b09dc09320b951c8782` | `87780e97759de78253b01b1fd6d53e3efdbdbbda` |
| `benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-final-2026-11-24-n10.json` | 17001 | `20cf89432b53f5776217e8d072dcb71afcda997ed3d4ed6ab611bc0d5cbcddd0` | `69afaad1990dc997e4c56d91b5e0963d35eb7249` |
| `benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-final-2026-11-24-n10.summary.json` | 1949 | `807f1aa554c2ddc1e8114ba2323808cac3b003f5a60dea8727faf54e830bb390` | `73b67b81ca4f87131745fb12f36f4c6fe87bed17` |
| `benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-final2-2026-11-24-n10.json` | 17002 | `ba1cd904b3571ecfb0af30d506cabf303e65e8cad12585b9ceb33a2176d391c4` | `9beaa2f9883f9d36d2b069cceb418647af63d0b3` |
| `benchmarks/interactive/results/results-paired-download-mcp-vs-agentbrowser-final2-2026-11-24-n10.summary.json` | 1937 | `a65ac7bdde783b223245409ea520205b03dfdac44c589219c6f66acfa4cbd1d1` | `8d99fd51b276aa04e6d090f96b9807c0aeddd5f0` |
| `benchmarks/interactive/results/results-paired-modal-2026-11-24-n10.jsonl` | 10993 | `5699cd450c80a96906722a42bc2bef8b3d07648b293528ea2bafff03e13a8ca1` | `484987a2044b6d2366512e0e1607b9ba3eb399e8` |
| `benchmarks/interactive/results/results-paired-modal-2026-11-24-n10.summary.json` | 1931 | `895304b671445fb50aab18ff4c29f5dde35b48a5c9a1928dabd64045a9c7ec78` | `d7e2b874d65d7519fcee5ec8e369c493c383cce0` |
| `benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-2026-11-24-n10.json` | 11076 | `42eee2f284b1c5343e9e161f6c8fda16f059f8b4178e741c9047e974aaa3cf0b` | `83302f704577da90f4acff59492e4ca586a02912` |
| `benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-2026-11-24-n10.summary.json` | 1951 | `e60cebfa96c7da6434acf44254637908a4bc3ebd9e09d5cc5cdab193d0703801` | `059254c8485fc33a080c0b08c1442ca8ebf21a5c` |
| `benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-final-2026-11-24-n10.json` | 11078 | `f0d1873e4ce995d3e648e3c835926649e2176102f6db1deb420e386e49d13b8b` | `b69fedfe5d33f31f0ad7abe139e352c3aa11b29c` |
| `benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-final-2026-11-24-n10.summary.json` | 1942 | `698cf00fba53aff52adfe2d983d7f3eb8212e17212cc5db3cee55ca4b859e9b8` | `2aa663b2fa8689182cb2d641c1b83816ae9a5434` |
| `benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-final2-2026-11-24-n10.json` | 11079 | `a4a66afd834ac6d71b976e16952943be2a7d8796fe222d35241109095d47aab2` | `7ab4e0cf7d17658036b5ab302eb80fd9adbb85c7` |
| `benchmarks/interactive/results/results-paired-modal-mcp-vs-browseruse-final2-2026-11-24-n10.summary.json` | 1933 | `1cf8684fb4813592cf0fc5451644091ea8438a48fba0f3148cc61216d9719ea3` | `81cb242e10da8ad0044d8e01bec64882dd4ec719` |
| `benchmarks/interactive/results/results-paired-tabs-2026-11-24-n10.jsonl` | 15032 | `da10df04a59e415224a9564f274d3f510130e3fc415df85f281cd952ac6e2f51` | `d3fe82dd8c9efa4a97bbd785078bc36403d15f2a` |
| `benchmarks/interactive/results/results-paired-tabs-2026-11-24-n10.summary.json` | 1923 | `a7cac7b90bc91b5d8785f35138a29b7c95c54ab21ebb495a66cb569999c03919` | `67618c1ee52e2f733f446b1200ef7abc60771c50` |
| `benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-2026-11-24-n10.json` | 15108 | `07e3b36b196e2c8b4d735db1c5a03dab92a2b913cb82a8eb1b4b374675fffcfa` | `4840f060a8499a012056a85580fd4fe2a7dd2451` |
| `benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-2026-11-24-n10.summary.json` | 1943 | `f0de9ef4c7386c39a8be6da48ce9bdecbd3b3d66a52e4f0fac0a802f1943d10f` | `ea57373c95e388b3a3391ed6e0900d326b134c77` |
| `benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-final-2026-11-24-n10.json` | 15115 | `c555cf9aa0abbc29471b3560a2f6a2685991610caa7155663097c98a499feafe` | `354773022438c17f72cd0b27cc93ba35dea8677a` |
| `benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-final-2026-11-24-n10.summary.json` | 1936 | `e29da03b0d08ada74f4085e4a3c3fd1691f5dbcfb186db51fbd5e85ead9cbd10` | `ab2a628d96a6d4788c0c691200a1346b771d0575` |
| `benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-final2-2026-11-24-n10.json` | 15108 | `2768ae3ab9f271c31b28a8cfae998cbfa48ce2bc183abeb166a180b76f1bee62` | `36906b277e5cc9c742ca6f94586a4559f238336b` |
| `benchmarks/interactive/results/results-paired-tabs-mcp-vs-browseruse-final2-2026-11-24-n10.summary.json` | 1925 | `459c69e24685646c7caf7fa256c6a156fdb34ce76480cab4d06f25698dc1dad8` | `ecec9cb18045e1497676bd87747f5d39652b0b28` |
| `benchmarks/interactive/results/results-paired-wizard-2026-11-24-n10.jsonl` | 10432 | `2e4cff38868cf900e9602c9264ca2c397dc00b2e823b16c663b0ac6efd1af199` | `b6a05b916e93fc451adc5ca50e81d35333b271b8` |
| `benchmarks/interactive/results/results-paired-wizard-2026-11-24-n10.summary.json` | 1939 | `c8c8f596c44497d70f785d99451a0be7fc4be4f39d0103b20c496ff1565562a6` | `0ac8da0ab6a7eb48df88666e5ec5ae23f91fb187` |
| `benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-2026-11-24-n10.json` | 10522 | `2f1d8e0afc32fb36a511fa6b3661c4b85c53c09c12b04623145e4f551da4034e` | `515c84177067520afc30da8f00a66bd2fe9586c5` |
| `benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-2026-11-24-n10.summary.json` | 1947 | `03d3518eaf17240a8411e7cd0e3f673fb558797f4cabc72956211a1c7556cac0` | `ab42f1d07bd9e727ed1531556f87efbacd0158a8` |
| `benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-final-2026-11-24-n10.json` | 10519 | `b9ab5ce659dadc19b13a4a3b6331583d3c16f212c89d57b67aee137d470459a7` | `86a540987c4bbbb9d0589dc549c74dd379d1eb3d` |
| `benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-final-2026-11-24-n10.summary.json` | 1945 | `d7ccdf5e29c83e1278b31b8e1d2b1d3e545d2ea2d0e5d00070f541f365dc354a` | `83114d6e703a59e5d9b531dfb059fd264e12e616` |
| `benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-final2-2026-11-24-n10.json` | 10525 | `90b2a36d1fab2ea1d9680c1e556a8f1ae648ac2c6f7f1e02ecb1a68cc88eb4ef` | `bdaf5293c54f957d9b5edb0947c8c19e9532fae2` |
| `benchmarks/interactive/results/results-paired-wizard-mcp-vs-browseruse-final2-2026-11-24-n10.summary.json` | 1935 | `512e3dba70a96992f36607ad7af5610cab3af81388d21cb6060642e6a906754e` | `a2e039b06015b2409b2a8422b968306780796ee1` |
| `benchmarks/interactive/results/results-warm-2026-11-24-n5.jsonl` | 14295 | `33dfdf4cc96d4127504d2b8486bc3eaffe5a475129ba029f74879002f53ab9a7` | `07ef528f330e091da0cbf78278c74caeb536fd99` |
| `benchmarks/interactive/results/results-warm-2026-11-24-n5.summary.json` | 1503 | `e6383f1c4a86ad38d02a1a20395a2eb175fa753f238f26a0f8bd8b097b4a8382` | `5d8d5c135579a2cdbdab4be195e1f775f64fe7e9` |
| `benchmarks/interactive/results/results-warm-4arm-2026-11-24-after-popup-n3.jsonl` | 33891 | `67b8397285bc5bcec514c885899cda604ba320bb5a90264f17e1334cb2d6def0` | `8472d14c7f96dd6c4efc8c8354436e6072bbee7f` |
| `benchmarks/interactive/results/results-warm-4arm-2026-11-24-after-popup-n3.summary.json` | 6315 | `2f85fd9c116aa7cd7054a28b110e8689ee336bd777d9111ed20f1d7d843c07de` | `5ad602415c6a231a8b18a802168cc07a584ef611` |
| `benchmarks/interactive/results/results-warm-4arm-2026-11-24-n3.jsonl` | 33928 | `0142262e2007135e2039a3e1040f818ed9e2e856d1f0513ae0053cb1974e03c6` | `04cf22a1802e55fa9c6209b7f508a7fb966f800a` |
| `benchmarks/interactive/results/results-warm-4arm-2026-11-24-n3.summary.json` | 6317 | `39e4168c7b96c7a0ab3aebc4172f8f3eb7615b668163b6a750f6de8395cb7d2d` | `75218d61a95d1118c5a05a60433d3a71219ac3a6` |
| `benchmarks/interactive/results/results-warm-download-wakeup-ab-2026-11-24-n5.jsonl` | 20312 | `6cf5976708825a3fcda0760bfc456ea854c225cbff14086d34b4d564543398eb` | `9b7282c65e0f4c579bd7db64c7bad70f8466e722` |
| `benchmarks/interactive/results/results-warm-download-wakeup-ab-2026-11-24-n5.summary.json` | 4240 | `89d57c138700e17945e5511da6215764d6cd06daa29c26db20a5695e5a2dc5b3` | `b0ec04567161764bef9c4661ec5cbcf9ad9020ca` |
| `benchmarks/interactive/results/results-warm-fast-selector-final-ab-2026-11-24-n3.jsonl` | 18627 | `1b2596abf130ab8a8b5b7ae5931482ada721a3c48b3d5259b195da4aefc95794` | `01817b709c36ea1ebfddd4876c27e0f25c9b5577` |
| `benchmarks/interactive/results/results-warm-fast-selector-final-ab-2026-11-24-n3.summary.json` | 3054 | `2b064556d3164ba1cddd3f63b717e9f37b100aa13f3daf739cd763cabc6c7a1d` | `bf39184c2a2b1eaca737c836acc2dc0e0a39cda3` |
| `benchmarks/interactive/results/results-warm-fast-selector-final-all-2026-11-24-n3.jsonl` | 43426 | `2d703774f8d9de5f2766209ece1028f69cf9090f678b18c3db8ddb76866c2396` | `2471eeab937c68abdd1b88a9e8bfce9b1f1350af` |
| `benchmarks/interactive/results/results-warm-fast-selector-final-all-2026-11-24-n3.summary.json` | 7687 | `a3bd25b6c5a56e0d7f0537e3f1923626f566404a27bb872177490ccb08dd7ba9` | `e35dc7c5b5037fc340fab57655acfcaef2e8bbfc` |
| `benchmarks/interactive/results/results-warm-five-arm-final-2026-11-24-n5.jsonl` | 76098 | `23e5c5059e1dd60b7d2c1f0399405862a39674340c372e7ca7e950d84b8774b3` | `83f5a7720416360b42848405ba2bece435016623` |
| `benchmarks/interactive/results/results-warm-five-arm-final-2026-11-24-n5.summary.json` | 7683 | `8b3b320c5e1a9dc885be926b50abb72e9eb917d88ad270ea9573dbee095654e5` | `76a4518bd037ac66c7de5cbbc93f2b9c5dea625b` |
| `benchmarks/interactive/results/results-warm-five-arm-final2-2026-11-24-n5.jsonl` | 76077 | `fb7afdfc2ca722aa0fdccb4d09eeaf597de1fa3a8a36140fbfdcda8c83336a2d` | `6e0452e2f5368ed3a60b3983a12bd47ef7de6561` |
| `benchmarks/interactive/results/results-warm-five-arm-final2-2026-11-24-n5.summary.json` | 7673 | `be688d97b5dd8a12a8c855ea93e8dc1c9316d3c72e3af067bd37cc913608ef84` | `f270ec0061d647fc21c040898d92fb147105960b` |
| `benchmarks/interactive/results/results-warm-five-arm-lean-2026-11-25-n5-final2.jsonl` | 88643 | `9da83e0c3b830db8287fa17fa140cb419aaaa455dde8400482f2a57faa1c4d32` | `cce15bc8943d871c1e93a328b0bcb3d0f5b419e9` |
| `benchmarks/interactive/results/results-warm-five-arm-lean-2026-11-25-n5-final2.summary.json` | 8011 | `d8d2a236bb9a73a7e6cb5ccf68a02df254237ba9a3acb42beba481d727c8d202` | `305ee850cb17c4b17cf95747512c887ea0856555` |
| `benchmarks/interactive/results/results-warm-five-arm-mcp-2026-11-24-n5.jsonl` | 76089 | `d6b69ee737bb9c16d578349047418781cbf860507f0942ef9eb2402433b85a3c` | `a5e4b7ede922e8155a33dce11db5fa83eb21b00f` |
| `benchmarks/interactive/results/results-warm-five-arm-mcp-2026-11-24-n5.summary.json` | 7681 | `320b4c2226ac9a49a64e9486dfc181180b8c99719f15dcc160177610fa85d78d` | `fd32a6f985e96985b318ecdbeb4e78ecec25d69a` |
| `benchmarks/interactive/results/results-warm-five-arm-tabs-ready-2026-11-24-n5.jsonl` | 73743 | `bd7cc8e10973bf57621b0cb71d77cb3281bab88685decac3ef1cf43f4326560f` | `dc7c339a9e665f4749f170c50a52044c40f267e3` |
| `benchmarks/interactive/results/results-warm-five-arm-tabs-ready-2026-11-24-n5.summary.json` | 7667 | `b926c4cd380ecedc1ab0623b88b30e1360ae1f850c3b6759ba1551fca8973365` | `d2a6fa824618315cf8d36d15666d90a0e2689b99` |
| `benchmarks/interactive/results/results-warm-tabs-direct-cdp-2026-11-24-n5-paired.jsonl` | 8834 | `1e14ee06c72f9f17fba5b9fa9031f34016b9c8a0494ff5011a1116c84dc0ab58` | `c87f331899828b53870ce9d0dc464826280feac2` |
| `benchmarks/interactive/results/results-warm-tabs-direct-cdp-2026-11-24-n5-paired.summary.json` | 2766 | `7233defad2dff3526e8ad15e3b2b576c21eb2981dc7271ae95b1b6b1a552a275` | `fade8d769f4b80c7b2917b8430cb96049c4ad68c` |
| `benchmarks/interactive/results/results-warm-tabs-five-arm-direct-cdp-2026-11-24-n5.jsonl` | 19072 | `2d37a2d135d7c6fee8b24478df0385979dbd4e2be288131f735c7faf390421e4` | `6c10fae4f4c433801e40f67fa23d1c2d4b8ed479` |
| `benchmarks/interactive/results/results-warm-tabs-five-arm-direct-cdp-2026-11-24-n5.summary.json` | 3844 | `3eef876870505a806d6e1b02c4eae30bd409eaf8fb233b6d2202aa88f6e2cd35` | `8cf67843454da5325df6fdf340a32846d6c7964f` |
| `benchmarks/interactive/results/results-warm-tabs-step-baseline-2026-11-24-n5.jsonl` | 7146 | `b1fc5911e6ac7ee7c1ba94e5b1cb3097a63e1b79fe5d4f8f755aa2fa904c58ba` | `8dea2ac73aab17259147c150d8f200923855210d` |
| `benchmarks/interactive/results/results-warm-tabs-step-baseline-2026-11-24-n5.summary.json` | 3476 | `ad126219318e32b1fd8be193521d8704a1cf4f605a6d179763537783bf16dd53` | `64fd7d772888254b33ae21180f3fc18929457ce1` |
| `benchmarks/scraping/results-2026-11-23.json` | 6595 | `cc79e4eec8fb1ccc51ce0bd749de89f5cf6826f3acf06f22ea2c721cbc997423` | `0b894e0aa7e9346c0ba5e5318a7ee769daa2c768` |
| `benchmarks/scraping/results-batch-firecrawl-2026-11-24-n10.json` | 2418 | `194d794ee5bbf5670fdd4d876e4ac8f8ed7286cf4d29dd10f4d3107ddca7d237` | `0a0a835a8dc426655b6f2146c56115835a21a440` |
| `benchmarks/scraping/results-batch-firecrawl-2026-11-24-n10.summary.json` | 1104 | `accf5f4df2aa8637c6ae7cecac34b8f0596d6ac3e388a59282959420f77e33e8` | `473fc4a75d373b468769e94cbe0ff84de75bfce6` |
| `benchmarks/scraping/results-cli-http-batch-2026-11-24-n10.json` | 3049 | `eb9e03936a6b0ea8d11d221d9400957fcbf5445d267d4469520b284dc90097ea` | `ba7b45acf0967d6162232959c2c312f71eecd97f` |
| `benchmarks/scraping/results-cli-http-batch-2026-11-24-n10.summary.json` | 758 | `3bd7e67159063474537bcc855c6c674470f5e14ad2b621f164967db2ae07e970` | `7fffa7aa58c68738e445049e53cb700a0c0cc520` |
| `benchmarks/scraping/results-fast-static-2026-11-23-n10.json` | 7575 | `4103f50ec5c5a680cd70697c1c3578d2956e13eb9fbd954adbdb913050509d59` | `a1aa29eb8718efbfa1786f6585d5aa623e0469fe` |
| `benchmarks/scraping/results-firecrawl-2026-11-23-n10.json` | 11863 | `705d852ab01fbce5de63a0e97ba2db31f9c0245801f991b4063e540d77e89eb6` | `369a5325d3a4cdbf98d4c1fbfb78ff78680c1d7a` |
| `benchmarks/scraping/results-firecrawl-2026-11-23.json` | 11793 | `32d09932cb85244179cee87538f0a97c0f1aab24aba1d1108f9cf136c2ff0e2c` | `7f0d516644e7c89fd7a9af0698e208839f212ed4` |
| `benchmarks/scraping/results-firecrawl-all-2026-11-24-n10-interleaved.json` | 7871 | `b027f876f8a8d0688eaa91836fc6d842a0e78360fb84544abd6b395b61198757` | `0a90b1013a8740618c28084280143efb77311a52` |
| `benchmarks/scraping/results-firecrawl-all-2026-11-24-n10-interleaved.jsonl` | 34845 | `aa055e50c3187539b736e42e1e25730265a8d178b94c386dc5b18e60626d65e6` | `a90727e63a21a0b6383632b889c4628201fad801` |
| `benchmarks/scraping/results-lightpanda-2026-11-24-n10.json` | 12545 | `28114188cde3311a9462f82ef00683f1123e2fd645aad0b43ed7b5885e0de0c5` | `09083386ba8976de0b5c6daf97daca1dc01c2a96` |
| `benchmarks/scraping/results-lightpanda-after-n10-interleaved.json` | 5485 | `9ee2487954deb33f21619955749ef655eeb57ee78f7b35a4c01fb4ea6230c52f` | `099797da58c4dd64370baf3af8774b83dd7acf43` |
| `benchmarks/scraping/results-lightpanda-after-n10-interleaved.jsonl` | 23240 | `2779091d80a5c2379dae6033cb8655ee6afec55758d71ffc72f4ae99d994921c` | `093ef62c5248c9fe72ff1ffff50bc01944604fa5` |
| `benchmarks/scraping/results-resources-2026-11-24-n3.json` | 14310 | `cbcb7552a62fcfa33c484b1c43df0e886d34f66b1995e16aa0505ded1c4cf6c5` | `b9e81d3d931965e612e33cce29cd35f828b6be6f` |
| `benchmarks/scraping/results/results-ab-asyncowner-A1-2026-11-24-n40.json` | 3658 | `940ab66818ec74bf6a0bc3387f98bbfe38699cd2bd6c825f317d62be094e2d66` | `a99557881a017bfef674d3aa9a1525ddf6a2a5a0` |
| `benchmarks/scraping/results/results-ab-asyncowner-A1-2026-11-24-n40.jsonl` | 54865 | `782a5404b307bc60b8d4fb9c63642d462609a391f78a264025d825a2705f853d` | `8d7c0c52926abc42ba776f44a12b3cd3e4b01d34` |
| `benchmarks/scraping/results/results-ab-asyncowner-A2-2026-11-24-n40.json` | 3653 | `9e8671018b588dca2b6f86529820a1f0ec49a836b0f8aa358d760a9bdc9ab204` | `886840d3bad922d3662f342c270188bfefb850fb` |
| `benchmarks/scraping/results/results-ab-asyncowner-A2-2026-11-24-n40.jsonl` | 54850 | `faf12fb1b8ca27d5231349f9e3d3167df877d708eed49ad6e5e95172cc0f505b` | `923a5c129ff7a1d6750b76f97beb3844470cbd30` |
| `benchmarks/scraping/results/results-ab-syncowner-B1-2026-11-24-n40.json` | 3649 | `74873ff66647b6d897299d912b89f2817949b227c5adf756b83a369900cdf799` | `c059e6878382fad7baa0cbbf3e3aeca56714dcdd` |
| `benchmarks/scraping/results/results-ab-syncowner-B1-2026-11-24-n40.jsonl` | 54834 | `a571ad7450e163a5e2f6a2b10e26de5cf93e19eaab36e09454a028cb22bfddb0` | `e9acfdfee5d06974cf5cfdd76e25de0614ec998e` |
| `benchmarks/scraping/results/results-ab-syncowner-B2-2026-11-24-n40.json` | 3657 | `58596941593674c48391e2a1af42c8c5d2caf2e7a95780020386b1dcca6a8343` | `2ce6ca714e00aa3931619bb40598c9de6bf21b3e` |
| `benchmarks/scraping/results/results-ab-syncowner-B2-2026-11-24-n40.jsonl` | 54847 | `e6119469a49892133d2dbeef5aa1d1db903cee12cd1b306c19fc6d3f4df817a6` | `cc44fef81a1187dad2052837d96d85174d35c682` |
| `benchmarks/scraping/results/results-batch-firecrawl-final-2026-11-24-n10.summary.json` | 1105 | `cb9da4cb275a2464a1ab716987ca11cc1a1ac1f3b51c31e41fc1b43b412b2a8f` | `6f06b5a938094164eaf5314db61e842f048333f6` |
| `benchmarks/scraping/results/results-cli-static-final-2026-11-24-n5.json` | 2576 | `4e53921b025924b2c7208a6f498a44155f1b287024c04b3e17a0703569547d67` | `b9c4a1c8f00617675226f0f34d6c83490916c9d8` |
| `benchmarks/scraping/results/results-cli-static-final2-2026-11-24-n5.json` | 2574 | `dddd497896ef8d8fcfae897715a68cfaf5d66d490cba4163d8718db3e7ab6b30` | `69164f3ff0b5c7ef60753c2788945b35d1718787` |
| `benchmarks/scraping/results/results-cli-static-resources-2026-11-24-n5.json` | 2579 | `157773d1aa63ce9be991f1529b12783b52948e8a5f35d1887859368654f5998c` | `c3dcb487a91e6cfff4133af836fea2bfb8dac38e` |
| `benchmarks/scraping/results/results-firecrawl-final-2026-11-24-n10-interleaved.json` | 7539 | `f2b74cfe276461e19de2f2c577389d0d3cd19124d715a3ed00d9d1af57d3d643` | `b7c432048f059d2dcfd218276fc7f46a2b7e5424` |
| `benchmarks/scraping/results/results-firecrawl-final-2026-11-24-n10-interleaved.jsonl` | 34481 | `bc7a7b8621069141b3f8e0956f581fe4f693c9feccfddb6b45767e2d2b24bf44` | `68c8cce34e0edeef1ec1b30ab71dd8c4f2a10a4b` |
| `benchmarks/scraping/results/results-firecrawl-final2-2026-11-24-n10-interleaved.json` | 7552 | `4355e33ad2cf953ac72a01608dab1b0401e5cf703093395a234f5e57dd8ebdbb` | `586de3fd690f3e45e80ce9c0d410ab0c88065849` |
| `benchmarks/scraping/results/results-firecrawl-final2-2026-11-24-n10-interleaved.jsonl` | 34518 | `f74d7eab37f374a4f838c7998a322d558711238beba711c5d1a1bc662583a922` | `80d4c8ee3cc2f0d3475a9100a54d834bfd7ba1ac` |
| `benchmarks/scraping/results/results-firecrawl-final4-2026-11-25-n10-interleaved.json` | 7545 | `3bcfcf4d1fd750efa2d9cc33ab4c9dc97b51eec32e28efe48e9fbd28958b5c42` | `336327f0c784c2325456b133fdcdf9e82e08b8ef` |
| `benchmarks/scraping/results/results-firecrawl-final4-2026-11-25-n10-interleaved.jsonl` | 34498 | `427bb96ac47f3a3ea652d28347f987d5b800a9a6f89ade59903c03346e58e532` | `34a00b20759fafa6a07399713f1ebe97dbe76557` |
| `benchmarks/scraping/results/results-firecrawl-final5-2026-11-25-n10-interleaved.json` | 7551 | `042f0c8292df008b985a0e426cc4b5e0c5ec28957423f5c6922986f62ebfd461` | `636613b11ff7d071a36b1cdbb9b74142fe7913c8` |
| `benchmarks/scraping/results/results-firecrawl-final5-2026-11-25-n10-interleaved.jsonl` | 34496 | `d491ea8c8dbf132cf16f6a3d12738ce90e1ede79b01be30393e279c3c34ec0a1` | `62e1b312e7a9c8af85fa5238ca7b7c837357e188` |
| `benchmarks/scraping/results/results-headless-shell-ab-2026-11-24-n6.json` | 2642 | `1d94d7bd55b934a60ef2c9227ea74cb4d7e8ea48a7ba8f83c54c2809b278b9b5` | `d6bfcf260b4c5bc05acddf291c5623cad8f1a76d` |
| `benchmarks/scraping/results/results-headless-shell-final2-2026-11-24-n6.json` | 4692 | `70c796df882e0be82b61842151dc1f90dac2a340da84f0854a2e429bd2f2f021` | `b9f2ce3799985090a7b587d8c03e292c667e2f04` |
| `benchmarks/scraping/results/results-lightpanda-final-2026-11-24-n10-interleaved.json` | 5320 | `79851030e896befcea634c6b70428221ca9b2ff59b54e21c1372cf05140f2b8e` | `0e96763a29a8ad3bf6ba41b71d0d5a88eaad7625` |
| `benchmarks/scraping/results/results-lightpanda-final-2026-11-24-n10-interleaved.jsonl` | 23081 | `0bb02d2d28c95393b519d80dcb8a307c94aab070da82393f9237e0a2a354469a` | `fbe21d3ac71485f191bbeacae196b3a886296265` |
| `benchmarks/scraping/results/results-lightpanda-final4-2026-11-25-n10-interleaved.json` | 5331 | `d9cd8c195822620a22443f0d1c1b59a81dbd588ab13854bc9a7b9f4a3027b034` | `3fe10ca379b480cf54d70758178d04f09a500767` |
| `benchmarks/scraping/results/results-lightpanda-final4-2026-11-25-n10-interleaved.jsonl` | 23100 | `3c9b239afee8f603b366755ede366ae2ada978813b8f301addd1d71966e6e344` | `a933d470326137c2f1f64f9ea25de73e672ebd03` |

## 4. Retired charts and scripts

| Path | Bytes | SHA256 | Git blob |
| --- | ---: | --- | --- |
| `benchmarks/scraping/plot_batch_results.py` | 5249 | `dfbdb6bbb90451143fbccbfc512967f03df807b5767a4e6541fc9a4b37604a44` | `182953426f6a912a56363f5c8e9a550b4ef6b0f7` |
| `benchmarks/scraping/plot_resources.py` | 4408 | `b8153c79ed7bc297d54ec4d59c6caf744efff792e36199e4b5d30efe1ac9b5ba` | `66f7155cea55e30a635b08b780edeac1e3916b49` |
| `scripts/annotate_historical_charts.py` | 3035 | `d3d66dab7545a528c82e9ea7c4e21b64d0d15200735f1c8972705a6a91331b63` | `faf7fc1b2ff48f7780bd38aec469cbec1ea4d167` |
| `benchmarks/benchmark_comparison.png` | 130257 | `5d5ce39ee283ae42ece625615178450d873ef56a1970a9e9b422a453cec94fd6` | `ca7b277d424d9f279b15cc442f641d5c076abde3` |
| `benchmarks/cli_benchmark_overview.png` | 161114 | `d8c3f8438d25d162db85cb8be6712b42e6105e2c0e5f106c1682170da1170d52` | `79f141e08eb0a76020fdc90d8adce231bc60bc3a` |
| `benchmarks/cli_benchmark_per_task.png` | 118210 | `90db71f3ccea642358ad788149d703f2a336f8078c2e92e92d6dd76440bb19ca` | `317aff440ba3fc4f4000e744970f858c8f96cf45` |
| `benchmarks/cli_benchmark_scatter.png` | 164455 | `c7fec908aa2eb678506286faa9087d0c3371fc589a47a10f1c7216c7cb35f6ad` | `2267f0bed01bacd3c2012cac6cbc2b3ee093f557` |
| `docs/images/benchmark_cost.png` | 109754 | `129d2c778909f1ae38604fc91bdda78f1660e67f25c1a66e00c5b3e2a4ca82b4` | `56b133ea5daef0d720953696109dd3a5c6596117` |
| `docs/images/benchmark_per_task_duration.png` | 115275 | `f2a983780ec17b7309549be645d6c4ea7abb84b2285d4bd28f6fd00012e03d66` | `eef38427acac943d8093834fe18acb4dbba236ee` |
| `docs/images/benchmark_per_task_tokens.png` | 134211 | `616277aea46a70151128b5fd2e23a1b3dfbb818c14ed0b81b84d6a23d4f489cf` | `9f5ba9e927d21c7b2d0e41ddf328ab6e6def45f7` |
| `docs/images/benchmark_response_size.png` | 107739 | `96d86e1f6f05991d0b97a7381eacf5df25ccddeca864d10d407392a618616e20` | `c733c314bc7bd97515de9c13d3d404e27e727c4a` |
| `docs/images/benchmark_token_breakdown.png` | 83734 | `986de497b4e5e3b8271d1de441afcc5a9bb5d799ab2203838addfa8bbd5ff696` | `f896417e53cdd5437c02612c2cea1e39004c7981` |
| `docs/images/benchmark_tool_calls.png` | 94195 | `c541ff5ad51661749a583bfbc29ea4280d7dd921e04481923c03523cf421b38d` | `1b0823ad737f634901556ea9e6f7a2aa6d170db9` |
| `docs/images/cli_benchmark_overview.png` | 161114 | `d8c3f8438d25d162db85cb8be6712b42e6105e2c0e5f106c1682170da1170d52` | `79f141e08eb0a76020fdc90d8adce231bc60bc3a` |
| `docs/images/cli_benchmark_per_task.png` | 118210 | `90db71f3ccea642358ad788149d703f2a336f8078c2e92e92d6dd76440bb19ca` | `317aff440ba3fc4f4000e744970f858c8f96cf45` |
| `docs/images/cli_benchmark_scatter.png` | 164455 | `c7fec908aa2eb678506286faa9087d0c3371fc589a47a10f1c7216c7cb35f6ad` | `2267f0bed01bacd3c2012cac6cbc2b3ee093f557` |
| `docs/images/native-interactive-warm-p50.png` | 97219 | `eb14310c7901d8fcde4f6753a42e167ddea85b32401eddfd4bb5e2e086c97bd5` | `6aa214a8ee13b6a410dfad2b3edfe101ec2ea85a` |
| `docs/images/native-scraping-p50.png` | 112798 | `47ef0382906d9df35e16e449afa13e01fbca7c351518feba7dffdcadc1b41c71` | `a842591dce899155f2d39edb33d0f5e24f4265bd` |
| `docs/images/native-scraping-batch-current.png` | 152311 | `02a781fe57ca6e800a6df11da64b29491c4287d4cd27d7e271d271a7daa0bf36` | `ebd89f84d556ee7c616ed4c99ec0bef521947670` |
| `docs/images/native-scraping-resources.png` | 158601 | `655ebd90a6aca0d2a4cc01721a00bc3da552f48e60369c6d1a7f5b6e8ec2ccad` | `37d19a88d352d5fbd1740346db24a6f7f0a21999` |
