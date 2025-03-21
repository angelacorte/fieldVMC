## [1.26.4](https://github.com/angelacorte/fieldVMC/compare/1.26.3...1.26.4) (2025-03-21)

### Bug Fixes

* change parameters ([90c4a52](https://github.com/angelacorte/fieldVMC/commit/90c4a52809014d3fc604012aef5f69ea8fad6b0e))

### General maintenance

* **release:** update gradle.properties .env versions to 1.26.3 [skip ci] ([63fea14](https://github.com/angelacorte/fieldVMC/commit/63fea1464abcf9bda6891dc6d5302a0c7920a454))

## [1.26.3](https://github.com/angelacorte/fieldVMC/compare/1.26.2...1.26.3) (2025-03-20)

### Bug Fixes

* change parameters ([e3fa36d](https://github.com/angelacorte/fieldVMC/commit/e3fa36d09c114bdb1334dfa76dccfa75a7fb8a0a))

### General maintenance

* **release:** update gradle.properties .env versions to 1.26.2 [skip ci] ([e8804ba](https://github.com/angelacorte/fieldVMC/commit/e8804bae28ab36c37b7460d62d886fddd6fd4537))

## [1.26.2](https://github.com/angelacorte/fieldVMC/compare/1.26.1...1.26.2) (2025-03-20)

### Bug Fixes

* remove deploy constraint in docker compose ([c1e5bfb](https://github.com/angelacorte/fieldVMC/commit/c1e5bfbca62b18a23735c48ac1bad4857964c39b))

### General maintenance

* **release:** update gradle.properties .env versions to 1.26.1 [skip ci] ([ac21860](https://github.com/angelacorte/fieldVMC/commit/ac2186097b70b45060f87d3f1209405a49bd31e2))

## [1.26.1](https://github.com/angelacorte/fieldVMC/compare/1.26.0...1.26.1) (2025-03-20)

### Dependency updates

* **deps:** add pandas dependency ([b6fb38f](https://github.com/angelacorte/fieldVMC/commit/b6fb38faf164a1d572812ee9f3ebabf2977ba70b))

### Bug Fixes

* change layers position ([9cb3962](https://github.com/angelacorte/fieldVMC/commit/9cb39626b6b5f1d8da4e3acbd65a68b8ff55eb5e))
* change max resource and success amount ([641ed5c](https://github.com/angelacorte/fieldVMC/commit/641ed5c3f73bee489f7c6b95bd870d48dfe34a02))
* resource management for classic vmc ([8ed3f2a](https://github.com/angelacorte/fieldVMC/commit/8ed3f2a16b06a02d8c12cbbc71e6cbabe15f0957))

### General maintenance

* **release:** update gradle.properties .env versions to 1.26.0 [skip ci] ([3a7e590](https://github.com/angelacorte/fieldVMC/commit/3a7e59050c65a9bb81bdc2cd4530428837219f70))
* remove old data ([82f77c2](https://github.com/angelacorte/fieldVMC/commit/82f77c2e986cc7e39f75d24e3e4af22bf31ac074))

## [1.26.0](https://github.com/angelacorte/fieldVMC/compare/1.25.2...1.26.0) (2025-03-20)

### Features

* add custom charts generator ([4e617d9](https://github.com/angelacorte/fieldVMC/commit/4e617d94c4ccb965f0425f360544a6eed68e14ba))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.13 ([8ff9bf3](https://github.com/angelacorte/fieldVMC/commit/8ff9bf3174e246ec5361cd13cafad4eaeac43009))

### General maintenance

* add optimized parameters for classic vmc with fixed resources ([4412917](https://github.com/angelacorte/fieldVMC/commit/441291755974997a78e6402d0c12c68e32e61f22))
* **release:** update gradle.properties .env versions to 1.25.2 [skip ci] ([d711971](https://github.com/angelacorte/fieldVMC/commit/d711971881c50aaa5cc4984623b2f3139f0e7333))

## [1.25.2](https://github.com/angelacorte/fieldVMC/compare/1.25.1...1.25.2) (2025-03-19)

### Bug Fixes

* distance evaluation metric ([6e92e35](https://github.com/angelacorte/fieldVMC/commit/6e92e35f14c3c965d52ccae867a4ab0d90c44314))
* initial nodes to arbitrary variable ([f6a74ac](https://github.com/angelacorte/fieldVMC/commit/f6a74acf13cd89a375c63a5692b4571bae40afff))

### General maintenance

* **release:** update gradle.properties .env versions to 1.25.1 [skip ci] ([6da9750](https://github.com/angelacorte/fieldVMC/commit/6da9750da10a0d102f46b4dae05c3bce3b2218ff))

## [1.25.1](https://github.com/angelacorte/fieldVMC/compare/1.25.0...1.25.1) (2025-03-19)

### Bug Fixes

* get parent as int ([bb3d5f3](https://github.com/angelacorte/fieldVMC/commit/bb3d5f3e75e611aed47eac2da14583b31f7d33c4))

### General maintenance

* **release:** update gradle.properties .env versions to 1.25.0 [skip ci] ([c2d9185](https://github.com/angelacorte/fieldVMC/commit/c2d91854aca37cfa2a33828d014cadb0a32e56c2))

## [1.25.0](https://github.com/angelacorte/fieldVMC/compare/1.24.0...1.25.0) (2025-03-19)

### Features

* add custom network interface ([edd493e](https://github.com/angelacorte/fieldVMC/commit/edd493ebeefd3db607fa4206dcd66fa080db8e68))

### Bug Fixes

* change initial nodes into smaller geometric variable ([b09e384](https://github.com/angelacorte/fieldVMC/commit/b09e38493fb73612577508616c3604a776bb5cef))
* increase tasksize ([f76e199](https://github.com/angelacorte/fieldVMC/commit/f76e199a46399ae3c48f9a2fead346102604134e))

### Performance improvements

* improve subnetwork diameter evaluation by checking whether a node is the father of another node ([cb93cb3](https://github.com/angelacorte/fieldVMC/commit/cb93cb3fa650bef41495747fb2082efed5ea7dcf))

### Build and continuous integration

* **deps:** update actions/setup-node action to v4.3.0 ([2de00b8](https://github.com/angelacorte/fieldVMC/commit/2de00b8ce012a2a30df86d86eb677e264f44fbd6))

### General maintenance

* **release:** update gradle.properties .env versions to 1.24.0 [skip ci] ([73c10c5](https://github.com/angelacorte/fieldVMC/commit/73c10c55a90b313902cca889c250889d8f69a50a))

### Style improvements

* improve code readability ([39d19e5](https://github.com/angelacorte/fieldVMC/commit/39d19e51d3a3323cd7a03583ab01825682c7e5c7))

## [1.24.0](https://github.com/angelacorte/fieldVMC/compare/1.23.2...1.24.0) (2025-03-16)

### Features

* add self optimization service launcher ([25653d0](https://github.com/angelacorte/fieldVMC/commit/25653d04fdb5af153f8aed42b522a1ea247e4155))

### General maintenance

* **release:** update gradle.properties .env versions to 1.23.2 [skip ci] ([366648d](https://github.com/angelacorte/fieldVMC/commit/366648d9162e0cfe735b4ce97e0800623bcea611))

## [1.23.2](https://github.com/angelacorte/fieldVMC/compare/1.23.1...1.23.2) (2025-03-16)

### Bug Fixes

* network diameter evaluation as per alchemist PR ([4b169c3](https://github.com/angelacorte/fieldVMC/commit/4b169c3c42769e4cf9cb272261c4919c7e80e355))

### General maintenance

* **release:** update gradle.properties .env versions to 1.23.1 [skip ci] ([baf2694](https://github.com/angelacorte/fieldVMC/commit/baf269416c6f9aeff0fa0d08ad6893c464316489))

## [1.23.1](https://github.com/angelacorte/fieldVMC/compare/1.23.0...1.23.1) (2025-03-16)

### Performance improvements

* optimized network diameter evaluation ([196c14d](https://github.com/angelacorte/fieldVMC/commit/196c14d80aefb2c87c4e6d20bb12b343c9e66d75))

### General maintenance

* **release:** update gradle.properties .env versions to 1.23.0 [skip ci] ([0ce6c81](https://github.com/angelacorte/fieldVMC/commit/0ce6c8110656a1409ca17ba86362258339069037))

### Style improvements

* remove commeted code ([bbaf903](https://github.com/angelacorte/fieldVMC/commit/bbaf9032a85d4a471a20f6b48e559ea42cdfbca5))

## [1.23.0](https://github.com/angelacorte/fieldVMC/compare/1.22.0...1.23.0) (2025-03-14)

### Features

* add faster alchemist environment ([1666fcc](https://github.com/angelacorte/fieldVMC/commit/1666fcc33ac581f2fb7155ec824c30b120110cbf))

### Dependency updates

* **deps:** add symmetric matrix dependency ([a12d8e6](https://github.com/angelacorte/fieldVMC/commit/a12d8e649eb4cf9f4a40a60c0286951608ef79f8))
* **deps:** update alchemist to v42.0.1 ([09567f0](https://github.com/angelacorte/fieldVMC/commit/09567f0d70a2160d243db758da739c8c783907b4))

### General maintenance

* **release:** update gradle.properties .env versions to 1.22.0 [skip ci] ([bab7c1d](https://github.com/angelacorte/fieldVMC/commit/bab7c1de0156978d0f2d5ff07b35a6e984f36523))

### Refactoring

* adapt deployment name ([aced94b](https://github.com/angelacorte/fieldVMC/commit/aced94b7f45fc0a5ca54ff47566a55fad3649d52))

## [1.22.0](https://github.com/angelacorte/fieldVMC/compare/1.21.0...1.22.0) (2025-03-12)

### Features

* add deployment to create nodes into specific area of two concentric rectangles ([753a0c7](https://github.com/angelacorte/fieldVMC/commit/753a0c7b34445a5024882301dfaacfa90a93b391))
* add local-resource molecule assignment ([4acde72](https://github.com/angelacorte/fieldVMC/commit/4acde72f159e661805ea79c3e51deb9b2f412365))
* change deployment into "outerRectangle" ([c884be7](https://github.com/angelacorte/fieldVMC/commit/c884be7f2a0b9d10f3eb2fc1555a89fe51b41058))

### Bug Fixes

* restore destruction of nodes in spawning logic ([5bccae0](https://github.com/angelacorte/fieldVMC/commit/5bccae0a36e09aa22733f8fefadabe54cbb0c574))

### General maintenance

* **release:** update gradle.properties .env versions to 1.21.0 [skip ci] ([e8abc82](https://github.com/angelacorte/fieldVMC/commit/e8abc8264b4557ec2e5d6482b9127500e0b1784e))

### Style improvements

* invert position success-resource sources ([1187dc1](https://github.com/angelacorte/fieldVMC/commit/1187dc1dd0cbe8a7f8470f71aced205842c2de11))

## [1.21.0](https://github.com/angelacorte/fieldVMC/compare/1.20.0...1.21.0) (2025-03-11)

### Features

* add exporter for self optimization simulation ([0191309](https://github.com/angelacorte/fieldVMC/commit/01913098e986528dfde08e11e104daca9050099c))
* add services for self-* simulations ([0b5a388](https://github.com/angelacorte/fieldVMC/commit/0b5a388721ab9ca13b10108ac40d713553a879f7))
* **build:** add batch override for self-* experiments on initial nodes ([9f26311](https://github.com/angelacorte/fieldVMC/commit/9f2631115e7f4c365066e1b4c1f969c01ded9ed8))
* initial nodes ad arbitrary variable ([b11328c](https://github.com/angelacorte/fieldVMC/commit/b11328c1fc33fbe7b7fc198737b2b08de79a0f85))

### Dependency updates

* **deps:** update alchemist to v42 ([46cbcc2](https://github.com/angelacorte/fieldVMC/commit/46cbcc245da510ec8dcf30f3e52f20a744eba6fa))

### General maintenance

* **release:** update gradle.properties .env versions to 1.20.0 [skip ci] ([7130461](https://github.com/angelacorte/fieldVMC/commit/713046124b0310da84362921fdbebf6e3164ac2c))

### Style improvements

* remove unused import ([a490cfe](https://github.com/angelacorte/fieldVMC/commit/a490cfe73ff9441fc1af844eafb5ca6c682e1311))

## [1.20.0](https://github.com/angelacorte/fieldVMC/compare/1.19.0...1.20.0) (2025-03-10)

### Features

* add assignment to molecule local resource ([013421f](https://github.com/angelacorte/fieldVMC/commit/013421fd282c2f0fc1b0cc4d69700f4f878b1e40))
* add extractors ([ba002b6](https://github.com/angelacorte/fieldVMC/commit/ba002b61b91e298c078ec529a53ba2eb9dd9f268))
* add subnetworks diameter extractor ([44840ba](https://github.com/angelacorte/fieldVMC/commit/44840bab56302f708b29bce3170e86d6137f8f42))

### Dependency updates

* **deps:** update plugin tasktree to v4.0.1 ([7ca5c32](https://github.com/angelacorte/fieldVMC/commit/7ca5c32704b8d7c4a0c4466ab31ace8496c6c62a))

### General maintenance

* **release:** update gradle.properties .env versions to 1.19.0 [skip ci] ([30e313a](https://github.com/angelacorte/fieldVMC/commit/30e313ad4495b56bda45988b028a9153ecd9c18a))

## [1.19.0](https://github.com/angelacorte/fieldVMC/compare/1.18.1...1.19.0) (2025-03-07)

### Features

* add exports to self repair simulation ([ba9874d](https://github.com/angelacorte/fieldVMC/commit/ba9874dc018310fded4720aeacbaf7bf9f062df6))

### Dependency updates

* **deps:** update alchemist to v40.1.3 ([062e533](https://github.com/angelacorte/fieldVMC/commit/062e533a40a1d689154fe2b569ed75be6c785a51))

### Bug Fixes

* adapt effects name to relative yaml ([b461e66](https://github.com/angelacorte/fieldVMC/commit/b461e6607c1796acfc480e6969851c0b686aae5a))

### General maintenance

* **release:** update gradle.properties .env versions to 1.18.1 [skip ci] ([8f4947c](https://github.com/angelacorte/fieldVMC/commit/8f4947c65a170e278cfc0278c9e329e7216f668d))

## [1.18.1](https://github.com/angelacorte/fieldVMC/compare/1.18.0...1.18.1) (2025-03-07)

### Dependency updates

* **deps:** update alchemist to v40.1.2 ([a0e7d8a](https://github.com/angelacorte/fieldVMC/commit/a0e7d8aaf31f7acb6e6107cf571da0ac405fab57))

### Bug Fixes

* rollback to objective function without weight on nodes ([2316c05](https://github.com/angelacorte/fieldVMC/commit/2316c0558bdb65295df0e07f0f0808f076cd3d0b))
* rollback to previous set of optimized parameters ([f0f7932](https://github.com/angelacorte/fieldVMC/commit/f0f79328541c260b5016c2945587bf7ff0f5a18a))
* update docker compose services ([bcb280f](https://github.com/angelacorte/fieldVMC/commit/bcb280f8c3fa533387c0841029ca2a35f98c17da))

### General maintenance

* **release:** update gradle.properties .env versions to 1.18.0 [skip ci] ([3eb2875](https://github.com/angelacorte/fieldVMC/commit/3eb2875401696388eeb7731122521a7da1d0d6fa))
* update fixed-leader data ([6e88ed4](https://github.com/angelacorte/fieldVMC/commit/6e88ed4c5021c0a78a8dab8223a84c9e283786c7))

## [1.18.0](https://github.com/angelacorte/fieldVMC/compare/1.17.2...1.18.0) (2025-03-06)

### Features

* update objective function with nodes metric power 2 ([3bef05c](https://github.com/angelacorte/fieldVMC/commit/3bef05c521683fdcc88eada00614449861aadbbe))

### General maintenance

* **release:** update gradle.properties .env versions to 1.17.2 [skip ci] ([c81381c](https://github.com/angelacorte/fieldVMC/commit/c81381c5b89f05e81bcba8c77d864874a44a4ff5))

## [1.17.2](https://github.com/angelacorte/fieldVMC/compare/1.17.1...1.17.2) (2025-03-06)

### Bug Fixes

* upate cutting field vmc with optimized pararmeters ([88c48af](https://github.com/angelacorte/fieldVMC/commit/88c48af0c5659ca44e1ed177781627fcefa2f6fb))
* upate field vmc with optimized pararmeters ([c5b345b](https://github.com/angelacorte/fieldVMC/commit/c5b345b6020df492a5b920b754c7a36dc3c13d5f))

### General maintenance

* **release:** update gradle.properties .env versions to 1.17.1 [skip ci] ([7aa1692](https://github.com/angelacorte/fieldVMC/commit/7aa16926f550f67d83e34687448e86d5f558821d))
* update classic vmc data ([adb456f](https://github.com/angelacorte/fieldVMC/commit/adb456f6e253d9c7a4c1342ca4bfea9c87fb150b))
* update nelder mead data ([ba7f2b6](https://github.com/angelacorte/fieldVMC/commit/ba7f2b6e6e0e9f8469d2a126f946952c970824ac))
* update optimized field vmc data ([bba287b](https://github.com/angelacorte/fieldVMC/commit/bba287b8ab8cd00f721d8246ab4a8db67a01fa71))

### Style improvements

* remove useless newline ([9c8cadb](https://github.com/angelacorte/fieldVMC/commit/9c8cadbc1c1f1ecbd76c5698c7e85c7116c87ad8))

## [1.17.1](https://github.com/angelacorte/fieldVMC/compare/1.17.0...1.17.1) (2025-03-06)

### Bug Fixes

* use apache geometric mean ([1c37d2c](https://github.com/angelacorte/fieldVMC/commit/1c37d2c94f0e21525e14e8d2c8af1a15aee60131))

### General maintenance

* **release:** update gradle.properties .env versions to 1.17.0 [skip ci] ([d07d158](https://github.com/angelacorte/fieldVMC/commit/d07d15864c5356931b6ab79ec10bed3e0ffef620))
* update services in docker compose ([b2494a8](https://github.com/angelacorte/fieldVMC/commit/b2494a8cb1b01fae90f1d15dfed8561c720c3c97))

## [1.17.0](https://github.com/angelacorte/fieldVMC/compare/1.16.2...1.17.0) (2025-03-06)

### Features

* move network metrics into proper object ([79540f8](https://github.com/angelacorte/fieldVMC/commit/79540f8d5b8cf4356a3e6ff8789434176c76df40))
* move termination metrics into proper class ([7cc380e](https://github.com/angelacorte/fieldVMC/commit/7cc380e94f28902359fa3031b383b8637d890311))
* update docker compose to run cutting simulations ([d242a20](https://github.com/angelacorte/fieldVMC/commit/d242a2068136eaee3944b406b7dfc4fa2d7c01d1))
* update optimized parameters ([d5c0ad1](https://github.com/angelacorte/fieldVMC/commit/d5c0ad17ed7bb459931a0b689210677b674fb256))
* update optimized parameters ([56ecda0](https://github.com/angelacorte/fieldVMC/commit/56ecda0e3923b45d8782fc642428deeb3bc6e039))

### Bug Fixes

* plotter ([cc57351](https://github.com/angelacorte/fieldVMC/commit/cc57351e398a643b7563622fccc22cd4ac1c2b7f))
* remove aggregator from diameter ([bbf7745](https://github.com/angelacorte/fieldVMC/commit/bbf77454c55cae9349b8bdef63bed4b4ff1bca33))
* remove double verbosity flag ([a9a774d](https://github.com/angelacorte/fieldVMC/commit/a9a774d2ac2d8b31089956a63f0006ba7bdae344))
* temporary remove charts from release ([a85d246](https://github.com/angelacorte/fieldVMC/commit/a85d246f257eaf80f600a0d73d0216a6af1fa718))

### General maintenance

* **release:** update gradle.properties .env versions to 1.16.2 [skip ci] ([e1ae2cd](https://github.com/angelacorte/fieldVMC/commit/e1ae2cdb6358156ae438bac178ec128eed8b4cf6))
* remove network density aggregators from data ([4a674b8](https://github.com/angelacorte/fieldVMC/commit/4a674b8cbdcb4c345ce8948b2fc64309b88901f1))
* upate optimized field vmc data ([f78350e](https://github.com/angelacorte/fieldVMC/commit/f78350ec08ef3387a44a616d06a27f231196ae91))
* update classic vmc data ([05c2853](https://github.com/angelacorte/fieldVMC/commit/05c2853a3fd89b64b8a78d5cc69818d8247ab55d))
* update nelder mead results ([bfbfc06](https://github.com/angelacorte/fieldVMC/commit/bfbfc06d6cc0d3ea465ad95eb5930b855832beab))

### Refactoring

* improve network density implementation ([31d3bce](https://github.com/angelacorte/fieldVMC/commit/31d3bcec7210b4c54ee6e3b45d4b895ae01fc115))
* move in outer scope network density ([ae640fa](https://github.com/angelacorte/fieldVMC/commit/ae640fa5aad1d1aaf52278964382186a8e68bd7c))
* move network density function ([f0dbeb3](https://github.com/angelacorte/fieldVMC/commit/f0dbeb3bec5d68e89ff8ec5f9159609fee78c994))
* remove aggregators from network diameter ([d62747b](https://github.com/angelacorte/fieldVMC/commit/d62747bed34ef03124cdd49d35b05b93e0e662e8))

## [1.16.2](https://github.com/angelacorte/fieldVMC/compare/1.16.1...1.16.2) (2025-03-05)

### Bug Fixes

* remove wrong data aggregator ([4ff3c62](https://github.com/angelacorte/fieldVMC/commit/4ff3c6247a71b6efd4d9ca0b39d5eefa2f4a4346))

### General maintenance

* **release:** update gradle.properties .env versions to 1.16.1 [skip ci] ([5d7f505](https://github.com/angelacorte/fieldVMC/commit/5d7f505fe1c9c7a3cf643549898278a064aaad2d))

### Refactoring

* implement network density without aggregators ([38ae60e](https://github.com/angelacorte/fieldVMC/commit/38ae60e7f5c7011f2cdd525f7f7f00480cd0ad36))

## [1.16.1](https://github.com/angelacorte/fieldVMC/compare/1.16.0...1.16.1) (2025-03-05)

### Bug Fixes

* change function to retrieve mean data ([7036365](https://github.com/angelacorte/fieldVMC/commit/70363656b6ab929d4f4caf88f1340a5a470955ea))

### General maintenance

* **release:** update gradle.properties .env versions to 1.16.0 [skip ci] ([6a6a00b](https://github.com/angelacorte/fieldVMC/commit/6a6a00b809fa7e23a995668b753cc2586d2d7145))

## [1.16.0](https://github.com/angelacorte/fieldVMC/compare/1.15.0...1.16.0) (2025-03-05)

### Features

* add ad-hoc simulation for the optimizer ([07608c2](https://github.com/angelacorte/fieldVMC/commit/07608c27de2612386608a47eea0ef5fa09c4b938))
* use fork join pool when fetching futures ([92a8c63](https://github.com/angelacorte/fieldVMC/commit/92a8c638714e5d7013c5ad6af0f422cd203e81aa))

### Dependency updates

* **deps:** update alchemist to v40.1.0 ([d496d20](https://github.com/angelacorte/fieldVMC/commit/d496d206c72acd2a23fb27e6088d77546d9cd573))

### Bug Fixes

* **build:** remove use of overrides for optimizer task ([0a73ff0](https://github.com/angelacorte/fieldVMC/commit/0a73ff098ef5e51f79a4ff99fad4b74b825c2f05))

### General maintenance

* **release:** update gradle.properties .env versions to 1.15.0 [skip ci] ([cb162de](https://github.com/angelacorte/fieldVMC/commit/cb162de305647bcb98a5926974848e63a642d25f))

## [1.15.0](https://github.com/angelacorte/fieldVMC/compare/1.14.1...1.15.0) (2025-03-04)

### Features

* add external volume constraints ([e46618a](https://github.com/angelacorte/fieldVMC/commit/e46618aff547f045f4cac93024d765b6ba2ec01d))

### Dependency updates

* **deps:** update alchemist to v40 ([6486988](https://github.com/angelacorte/fieldVMC/commit/6486988e923486ae514e178639b2a4d953b2c448))

### Bug Fixes

* calling coordinates on objective function instead of cache ([c80ecd7](https://github.com/angelacorte/fieldVMC/commit/c80ecd70745b2e9e1b08648744e2573b59d82af8))

### General maintenance

* **release:** update gradle.properties .env versions to 1.14.1 [skip ci] ([f1f9ad9](https://github.com/angelacorte/fieldVMC/commit/f1f9ad96f3961741a55a8e446c0dec28c2af2e57))
* upate classic vmc data ([5b7d629](https://github.com/angelacorte/fieldVMC/commit/5b7d62993966468c4cf4d8a8e48de3401b5a54a9))

## [1.14.1](https://github.com/angelacorte/fieldVMC/compare/1.14.0...1.14.1) (2025-03-04)

### Bug Fixes

* workdir path in dockerfile ([a71c1c2](https://github.com/angelacorte/fieldVMC/commit/a71c1c2d1348dbb8e3e9d9657290cfe258dd1949))

### Build and continuous integration

* **deps:** update actions/download-artifact action to v4.1.9 ([eac2a11](https://github.com/angelacorte/fieldVMC/commit/eac2a11fd93110dc11148509420f097b46122fe9))

### General maintenance

* **release:** update gradle.properties .env versions to 1.14.0 [skip ci] ([4d32c15](https://github.com/angelacorte/fieldVMC/commit/4d32c15d41f180bed53f6a78792d28bbcab29d96))

## [1.14.0](https://github.com/angelacorte/fieldVMC/compare/1.13.0...1.14.0) (2025-03-04)

### Features

* change docker compose to launch the optimizer ([48afa93](https://github.com/angelacorte/fieldVMC/commit/48afa93b897dae3542e0c9fd09019e2031f84f16))

### Dependency updates

* **deps:** update alpine docker tag to v3.21.3 ([785c201](https://github.com/angelacorte/fieldVMC/commit/785c20105d49ce2dae020c688bf710ee46d3d0c5))

### Bug Fixes

* remove experiment prefix in metric key ([b546c7b](https://github.com/angelacorte/fieldVMC/commit/b546c7b0a51cd0cd9c9a22a5374cdd33599350bb))

### Build and continuous integration

* **deps:** update actions/upload-artifact action to v4.6.1 ([ea12f35](https://github.com/angelacorte/fieldVMC/commit/ea12f356f2b2ef8c6d0012faedae2a199a93ea23))

### General maintenance

* **release:** update gradle.properties .env versions to 1.13.0 [skip ci] ([e73b5e9](https://github.com/angelacorte/fieldVMC/commit/e73b5e91f744fa9a387ee4d2c74c3278fae8bfe0))

### Style improvements

* missing tab ([9587c52](https://github.com/angelacorte/fieldVMC/commit/9587c523d311065a509214c54b56f2d73b1df511))

## [1.13.0](https://github.com/angelacorte/fieldVMC/compare/1.12.11...1.13.0) (2025-03-04)

### Features

* add check on difference result ([5cbfae5](https://github.com/angelacorte/fieldVMC/commit/5cbfae5f198c64a2f09bccd38e808636ea947157))
* add number of leaves extractor ([44d5c95](https://github.com/angelacorte/fieldVMC/commit/44d5c95482da325ce4e4c5b3cc774292d5d27312))
* add parameters optimization results ([fe71311](https://github.com/angelacorte/fieldVMC/commit/fe7131129abb5c89f10260a860b204ebce3993c7))
* add termination condition into task definition ([dfe14d3](https://github.com/angelacorte/fieldVMC/commit/dfe14d367661e03c646dfac4dbf2eb0997726e3a))
* create extractor for number of leaves ([cdbc1f6](https://github.com/angelacorte/fieldVMC/commit/cdbc1f66af512524fe8784fd33cc0284f5ad5866))

### Bug Fixes

* **build:** parameters override in gradle tasks ([a2ac652](https://github.com/angelacorte/fieldVMC/commit/a2ac652be294be94bc37dbe04d9e513905453a81))
* change time distribution into sparse node-execution ([8624e12](https://github.com/angelacorte/fieldVMC/commit/8624e123ca0309e34c04dec713e5f14abe9dd763))
* change time trigger for remove nodes ([49120b1](https://github.com/angelacorte/fieldVMC/commit/49120b11a50524d8c29b7ee2d237a7f912a0ed36))
* **ci:** add env variable to task ([bebadc6](https://github.com/angelacorte/fieldVMC/commit/bebadc6742d1ba8b4a55979f66c2ea7e9b56684f))
* data plotting ([34148b3](https://github.com/angelacorte/fieldVMC/commit/34148b3798ba3060685cc9250239affe02c77a9d))
* higher time samples ([1fe208a](https://github.com/angelacorte/fieldVMC/commit/1fe208a02588ad386352f2773faf9e48964aa6c5))
* longer time before cutting ([7809b3b](https://github.com/angelacorte/fieldVMC/commit/7809b3b8547bbe64d7730721d238977df8010369))
* objective function into 'geometric mean of the absolute differences' ([715cd6d](https://github.com/angelacorte/fieldVMC/commit/715cd6d287fd189d791b2b9809416d31720164df))
* remove duplicate terminator ([493b7b3](https://github.com/angelacorte/fieldVMC/commit/493b7b3cd9ab12f013c3ed9bf7c3500aab7d674b))
* update optimized parameters ([ce3f15f](https://github.com/angelacorte/fieldVMC/commit/ce3f15f58a89b8ce5863e0706fc306690ecd4540))
* wrong paths ([3fee205](https://github.com/angelacorte/fieldVMC/commit/3fee205dbbf2a3bb57b381000e56958d5e4d2243))

### Build and continuous integration

* restore charts creation ([bc8e58a](https://github.com/angelacorte/fieldVMC/commit/bc8e58ab6b00b767973389a7354b889fdc609f5f))
* update grale tasks ([2fa620b](https://github.com/angelacorte/fieldVMC/commit/2fa620b8469b91a414b7f577fc3c8bda8fc1074d))

### General maintenance

* add cleaned and mean data ([d48bbd3](https://github.com/angelacorte/fieldVMC/commit/d48bbd387066be71c08efeb826cdf248c6ad5ff1))
* **release:** update gradle.properties .env versions to 1.12.11 [skip ci] ([b503c1d](https://github.com/angelacorte/fieldVMC/commit/b503c1dbd969062fb8905f366199338300d90813))
* update data ([cb41cd3](https://github.com/angelacorte/fieldVMC/commit/cb41cd311afca5283e7113153c0f2ff3c970673d))

### Style improvements

* parametrize file name, path and max seed ([43445a4](https://github.com/angelacorte/fieldVMC/commit/43445a4cd2756ff37826d5c23a4c7581a0081845))

### Refactoring

* ACSOS' experiments names into relative self-* property ([9f197eb](https://github.com/angelacorte/fieldVMC/commit/9f197eb7ec2c7b166ee597eb25160aff10b143a0))
* change molecule name from myParent into parent ([fbf6afa](https://github.com/angelacorte/fieldVMC/commit/fbf6afaadf8f41491915a447febab92d1ca8f63d))

## [1.12.11](https://github.com/angelacorte/fieldVMC/compare/1.12.10...1.12.11) (2025-03-03)

### Dependency updates

* **core-deps:** update collektive to v16 ([#154](https://github.com/angelacorte/fieldVMC/issues/154)) ([42bfe50](https://github.com/angelacorte/fieldVMC/commit/42bfe50a52b1a5601a61f4bb9984a76f9f2be980))

### Bug Fixes

* rollback to working collektive version ([0d7d197](https://github.com/angelacorte/fieldVMC/commit/0d7d197db5dad21bbc0f19caf8803a5e5eb7962a))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.10 [skip ci] ([5b2bf81](https://github.com/angelacorte/fieldVMC/commit/5b2bf810b163a30d39dd5864373de83f28a231a8))

## [1.12.10](https://github.com/angelacorte/fieldVMC/compare/1.12.9...1.12.10) (2025-03-03)

### Dependency updates

* **deps:** update dependency matplotlib to v3.10.1 ([42c34cd](https://github.com/angelacorte/fieldVMC/commit/42c34cd79eef3c933b823b7f6aa5d32bcb553698))

### Bug Fixes

* add termination condition ([11bd42d](https://github.com/angelacorte/fieldVMC/commit/11bd42dd715301c3fb6f856e3b0acdae9e217c37))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.9 [skip ci] ([c8fc843](https://github.com/angelacorte/fieldVMC/commit/c8fc8434d5d9557ab56f47a382f013964775ebe6))

## [1.12.9](https://github.com/angelacorte/fieldVMC/compare/1.12.8...1.12.9) (2025-02-28)

### Bug Fixes

* add output directory ([0e4d287](https://github.com/angelacorte/fieldVMC/commit/0e4d287dff344467debefffabb46e07d6cd907a3))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.8 [skip ci] ([13590c5](https://github.com/angelacorte/fieldVMC/commit/13590c5252fe4041bf4a15a91a9434db118e7341))

## [1.12.8](https://github.com/angelacorte/fieldVMC/compare/1.12.7...1.12.8) (2025-02-28)

### Bug Fixes

* service dependency ([1210f08](https://github.com/angelacorte/fieldVMC/commit/1210f0895266034a5101018f08c31ca5addc0d58))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.7 [skip ci] ([f6ab7f5](https://github.com/angelacorte/fieldVMC/commit/f6ab7f52db4a01410d3daf697df436d2ad743396))

## [1.12.7](https://github.com/angelacorte/fieldVMC/compare/1.12.6...1.12.7) (2025-02-28)

### Bug Fixes

* remove wrong command ([678e581](https://github.com/angelacorte/fieldVMC/commit/678e581df8594727207e6cf865ebbfafa8b7fc15))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.6 [skip ci] ([1e074a4](https://github.com/angelacorte/fieldVMC/commit/1e074a4a421023e726ba3655db08215a27e5a09f))

## [1.12.6](https://github.com/angelacorte/fieldVMC/compare/1.12.5...1.12.6) (2025-02-28)

### Bug Fixes

* create two different services ([dbafb8c](https://github.com/angelacorte/fieldVMC/commit/dbafb8cba7eed623b33e26d9cd816bc0b385cb21))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.5 [skip ci] ([6e76aff](https://github.com/angelacorte/fieldVMC/commit/6e76affee52bb21d7a62a54b57363ae03e9e753a))

## [1.12.5](https://github.com/angelacorte/fieldVMC/compare/1.12.4...1.12.5) (2025-02-28)

### Bug Fixes

* docker image version ([1aa7b32](https://github.com/angelacorte/fieldVMC/commit/1aa7b3242895b63bcc03d8717f4b29969e20c5e1))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.4 [skip ci] ([750a4f1](https://github.com/angelacorte/fieldVMC/commit/750a4f1e954756ae5fffefb7401b1c6c646c9057))

## [1.12.4](https://github.com/angelacorte/fieldVMC/compare/1.12.3...1.12.4) (2025-02-28)

### Bug Fixes

* change docker file in docker compose ([29bced0](https://github.com/angelacorte/fieldVMC/commit/29bced0f86e89c18f8251b5a1a3b9dcc32bea074))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.3 [skip ci] ([ddd2dc4](https://github.com/angelacorte/fieldVMC/commit/ddd2dc4aec95b1d2cf42e552a855c316543047d3))

## [1.12.3](https://github.com/angelacorte/fieldVMC/compare/1.12.2...1.12.3) (2025-02-28)

### Bug Fixes

* correct path for target data ([607808c](https://github.com/angelacorte/fieldVMC/commit/607808ce379cc0dc59627b93e2b876a78f6789b2))
* restore yamls ([5b45b6e](https://github.com/angelacorte/fieldVMC/commit/5b45b6e0fbaed35f398dbebc071e260aaf969598))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.2 [skip ci] ([2a8b890](https://github.com/angelacorte/fieldVMC/commit/2a8b89088548142f739acdad558c45b07b3b7f62))

### Style improvements

* missing tab ([b155b02](https://github.com/angelacorte/fieldVMC/commit/b155b02c1ab2beafade2dc4fdf4b242c7ee54e68))

## [1.12.2](https://github.com/angelacorte/fieldVMC/compare/1.12.1...1.12.2) (2025-02-28)

### Bug Fixes

* docker compose with different gradle task into env ([c57ddda](https://github.com/angelacorte/fieldVMC/commit/c57ddda4ce2b6ac458a2334b08f51ea7ce908c10))
* objective function ([cca55db](https://github.com/angelacorte/fieldVMC/commit/cca55db4636645d8a28be85291a0ec6601300c3f))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.12 ([9af7417](https://github.com/angelacorte/fieldVMC/commit/9af74170d33e7956571c61f1ae37c0f30ec8d458))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.1 [skip ci] ([75294d1](https://github.com/angelacorte/fieldVMC/commit/75294d1951aed71434bf7a5fa3029116de52deb1))

### Refactoring

* export means also returns it ([455df5f](https://github.com/angelacorte/fieldVMC/commit/455df5f39b70207d9455dd88f36b668f6c479364))

## [1.12.1](https://github.com/angelacorte/fieldVMC/compare/1.12.0...1.12.1) (2025-02-27)

### Bug Fixes

* image name ([1609c2a](https://github.com/angelacorte/fieldVMC/commit/1609c2a195b9060f4a73a898e06eefd8d1cd56dc))
* remove wrong command ([b7c6d12](https://github.com/angelacorte/fieldVMC/commit/b7c6d1271cbd41a92ad084fcf5da82a69d5ae872))

### General maintenance

* **release:** update gradle.properties .env versions to 1.12.0 [skip ci] ([171de26](https://github.com/angelacorte/fieldVMC/commit/171de2619b0528763b482b592a9a1e50c1af9d8f))

## [1.12.0](https://github.com/angelacorte/fieldVMC/compare/1.11.3...1.12.0) (2025-02-27)

### Features

* add cutting classic vmc simulation ([01b4ea8](https://github.com/angelacorte/fieldVMC/commit/01b4ea8bef6202021e1de1d9cbc753ac5582f667))
* add cutting fixed leader simulation ([34bc467](https://github.com/angelacorte/fieldVMC/commit/34bc467949e953c825bbc33baab5880043718308))
* create export means into csv ([d44e482](https://github.com/angelacorte/fieldVMC/commit/d44e4823b298059613d270dad380cdea24afe1d7))
* update docker compose to run cutting simulations ([f81dd0c](https://github.com/angelacorte/fieldVMC/commit/f81dd0c4aee6610d47881998d88ff1b87af082a8))

### Dependency updates

* **deps:** update alchemist to v39.0.1 ([dee35a4](https://github.com/angelacorte/fieldVMC/commit/dee35a448167b5d38320cfc4f6498da5002699d4))
* **deps:** update dependency gradle to v8.13 ([50f6bbb](https://github.com/angelacorte/fieldVMC/commit/50f6bbb5e3be42ed8b9c20c7fb3305590855502f))
* **deps:** update plugin multijvmtesting to v3.2.3 ([11474ca](https://github.com/angelacorte/fieldVMC/commit/11474ca78f65a6321d18f12350775a5be85d924a))

### Bug Fixes

* check on null parent was a sorta of leader election, now check the real root ([bdbe2dc](https://github.com/angelacorte/fieldVMC/commit/bdbe2dc3c297ae12e8aa45c5c798b80a01c87bb0))
* correct directory name ([42634a1](https://github.com/angelacorte/fieldVMC/commit/42634a1c2e925496f310121a173e582a64779698))
* put files into proper directory ([375928c](https://github.com/angelacorte/fieldVMC/commit/375928c92f3e966474be1690dcf9cbd8d7169735))
* remove duplicate termination condition ([4225b80](https://github.com/angelacorte/fieldVMC/commit/4225b8085ebb41bdb53908540d16a0ee7e966dc7))
* remove duplicate variable definition ([23ea5b6](https://github.com/angelacorte/fieldVMC/commit/23ea5b6ce0d914f83b5385bafa773461786a4b95))
* remove duplicate vars ([5625743](https://github.com/angelacorte/fieldVMC/commit/5625743d62b0309895a3fc473a0aad432c89f43d))
* remove variables from batch ([7440da4](https://github.com/angelacorte/fieldVMC/commit/7440da4e8dd7313989152f43a53c12169c7a5368))
* remove wrong dependency ([3fd789b](https://github.com/angelacorte/fieldVMC/commit/3fd789bca79b7225206b148cb2a13531d26360e1))
* replace : with - in date time due to windows invalid character ([6e38ba6](https://github.com/angelacorte/fieldVMC/commit/6e38ba63c1c278a0107105b044f543d0f23a025f))
* use system path separator ([444a923](https://github.com/angelacorte/fieldVMC/commit/444a92351f4a9bda53d9511268e275b4ff891d88))

### Build and continuous integration

* create optimizer task just for fixed leader experiment ([0d7716d](https://github.com/angelacorte/fieldVMC/commit/0d7716d702c39d0abc5971df7a133d14d45c2779))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.11 ([9ebed40](https://github.com/angelacorte/fieldVMC/commit/9ebed40201e6d171b642a52a99cbae9fa80c4e8b))
* task to launch cutting* in batch on seed and origin position ([976b76c](https://github.com/angelacorte/fieldVMC/commit/976b76c4a7b584d17ec50e67de9cff4d3e1d3c86))
* write arbitrary variables into optimizer task override ([dbead51](https://github.com/angelacorte/fieldVMC/commit/dbead51a6b5f3cbd37923b0cd56e54d0a60cdf2e))

### General maintenance

* add classic-vmc and field-vmc batch simulation results ([51910e5](https://github.com/angelacorte/fieldVMC/commit/51910e54d80ef6c6943dd2543a647c61ee3e69ce))
* add nelder mead results ([af9c946](https://github.com/angelacorte/fieldVMC/commit/af9c946e682c143a99420b14bcfb8d870220e122))
* ignore data produced while executing nelder mead method ([cb1f51c](https://github.com/angelacorte/fieldVMC/commit/cb1f51c0ee6b21bf380c9c738881f0e33b4922c1))
* **release:** update gradle.properties .env versions to 1.11.3 [skip ci] ([aac85e1](https://github.com/angelacorte/fieldVMC/commit/aac85e16b569efe8b53d9a874f8865846bc32b4d))
* remove data ([ab49347](https://github.com/angelacorte/fieldVMC/commit/ab493472337aa6cbe2720ad7292dfe0e8bd866ad))
* rename directory ([8af8f58](https://github.com/angelacorte/fieldVMC/commit/8af8f58d244079b4958ccf78bff17392ef129c4f))
* smaller time interval to check for termination ([7047821](https://github.com/angelacorte/fieldVMC/commit/7047821e4b610c40674df84faa02ba451a6a8cf7))

### Refactoring

* invalid character for windows ([3bc7b0d](https://github.com/angelacorte/fieldVMC/commit/3bc7b0dad44f5f69cf8c10c94f9d9b53a841adf9))

## [1.11.3](https://github.com/angelacorte/fieldVMC/compare/1.11.2...1.11.3) (2025-02-20)

### Dependency updates

* **deps:** update alchemist to v39 ([935bfdc](https://github.com/angelacorte/fieldVMC/commit/935bfdc9a9e00b718ade1629d198f4254869c619))

### Bug Fixes

* dependency for docker service ([ffffbe5](https://github.com/angelacorte/fieldVMC/commit/ffffbe5d4f364b643e6369340af37633f0f7099c))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.10 ([ddacafc](https://github.com/angelacorte/fieldVMC/commit/ddacafcd8dcaf969b30e4009b3f381a4dfdb9fbb))

### General maintenance

* **release:** update gradle.properties .env versions to 1.11.2 [skip ci] ([a46fd0c](https://github.com/angelacorte/fieldVMC/commit/a46fd0c5eec7b92a03a376f6519638718e5d4624))

## [1.11.2](https://github.com/angelacorte/fieldVMC/compare/1.11.1...1.11.2) (2025-02-19)

### Bug Fixes

* push on docker hub ([0982fac](https://github.com/angelacorte/fieldVMC/commit/0982facb505956b72e3483b41e45f1997caec404))
* repository name must be in lower case for docker compose build ([5f31227](https://github.com/angelacorte/fieldVMC/commit/5f31227d4fec185d110b9dcda1b0c9a153402b95))
* typo in project name in docker compose ([489e950](https://github.com/angelacorte/fieldVMC/commit/489e95057f9f2c6401276112f47065c8f1f71135))
* typo in project name in docker compose ([cd2739e](https://github.com/angelacorte/fieldVMC/commit/cd2739e660caeb8fc2233485b80b7903dc5d4f2f))

### General maintenance

* **release:** update gradle.properties .env versions to 1.11.1 [skip ci] ([df95a91](https://github.com/angelacorte/fieldVMC/commit/df95a916708aff4d33f0329e727ecf2e1fabf6ca))

## [1.11.1](https://github.com/angelacorte/fieldVMC/compare/1.11.0...1.11.1) (2025-02-19)

### Dependency updates

* **deps:** update alchemist to v38.0.3 ([f3de305](https://github.com/angelacorte/fieldVMC/commit/f3de305182dc33697d61c36a9e20c37203324403))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.124 ([f59d636](https://github.com/angelacorte/fieldVMC/commit/f59d63608ed7e79b941a1ead25764ac18dcb9cc7))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.125 ([c18d848](https://github.com/angelacorte/fieldVMC/commit/c18d8483ecac8fc94b6946780e9d73f30fa5a014))
* **deps:** update plugin com.gradle.enterprise to v3.19.2 ([5ce8d90](https://github.com/angelacorte/fieldVMC/commit/5ce8d90afd8bdc3e5c2bfc45115f8013e524967a))

### Bug Fixes

* change project name in docker compose ([99361c5](https://github.com/angelacorte/fieldVMC/commit/99361c57af988358ccf797be9e7332ab9a54910d))

### General maintenance

* **release:** update gradle.properties .env versions to 1.11.0 [skip ci] ([5593bde](https://github.com/angelacorte/fieldVMC/commit/5593bde8ea6079576366b05c142583d078aeef5f))

## [1.11.0](https://github.com/angelacorte/fieldVMC/compare/1.10.0...1.11.0) (2025-02-19)

### Features

* implement termination based on stability of metrics for a specific amount of time ([4f50001](https://github.com/angelacorte/fieldVMC/commit/4f50001044ebfa17c1325c6edd1cc441ada17643))
* setup docker for parameters optimization ([2ce51c1](https://github.com/angelacorte/fieldVMC/commit/2ce51c11d0237fbd886558aa8ecf931b3db226c9))
* use termination stable for time ([af5e14b](https://github.com/angelacorte/fieldVMC/commit/af5e14b82210fa95b769ebeba0a9fe1f4b260db0))

### Dependency updates

* **deps:** update alchemist to v38.0.2 ([#158](https://github.com/angelacorte/fieldVMC/issues/158)) ([7bfd775](https://github.com/angelacorte/fieldVMC/commit/7bfd77599b9e89648791849595610bee0fc07d5c))

### Bug Fixes

* error on portainer 'depends_on must be a list' ([8a4fa66](https://github.com/angelacorte/fieldVMC/commit/8a4fa6638ff8a58bf07651109f94d5a237315561))
* missing class name refactor ([2d4302b](https://github.com/angelacorte/fieldVMC/commit/2d4302b6781e2b9470fc733f6e584630c2ce7896))
* missing sum on counter ([c4d95e5](https://github.com/angelacorte/fieldVMC/commit/c4d95e528f19a267e1bdc41712580ceac299f42d))
* reset counters ([ea6724a](https://github.com/angelacorte/fieldVMC/commit/ea6724a14efe6c4f29c189cb6c23784570302e7e))

### Build and continuous integration

* temporary disable check simulations docker ([4d9c18b](https://github.com/angelacorte/fieldVMC/commit/4d9c18b0d21a898e7bcb96ee61e417bde2491b30))
* temporary disable download charts artifact ([1b8de7e](https://github.com/angelacorte/fieldVMC/commit/1b8de7e6a81ae1008503c292426518789ea862dc))

### General maintenance

* **release:** update gradle.properties .env versions to 1.10.0 [skip ci] ([b300131](https://github.com/angelacorte/fieldVMC/commit/b300131fe7492ae2fefdf83b9e12f51d27ab39f8))

### Refactoring

* increase max seed ([e365ef7](https://github.com/angelacorte/fieldVMC/commit/e365ef7d2c500dfed170263d79675a23f0aef40d))
* increase tolerance ([90b8701](https://github.com/angelacorte/fieldVMC/commit/90b870168a5571aa73bb846afbec3111dd7ad8ec))

## [1.10.0](https://github.com/angelacorte/fieldVMC/compare/1.9.0...1.10.0) (2025-02-18)

### Features

* add function to determine whether terminate the simulation ([ccf6d05](https://github.com/angelacorte/fieldVMC/commit/ccf6d05fb3d92d02f13a8983ab603a5d24f48e41))
* add metrics termination condition ([6ecf897](https://github.com/angelacorte/fieldVMC/commit/6ecf897e99679021fe3b6887bf630355ae44e0b0))
* implement termination based on stability of metrics ([bf4f7e8](https://github.com/angelacorte/fieldVMC/commit/bf4f7e8444029be2f10f8257f9c64932914c1c35))

### Dependency updates

* **deps:** update alchemist to v38.0.1 ([#150](https://github.com/angelacorte/fieldVMC/issues/150)) ([b89bfd6](https://github.com/angelacorte/fieldVMC/commit/b89bfd63173ae8c974cc181978675e90142fa12e))
* **deps:** update alpine docker tag to v3.21.3 ([338029b](https://github.com/angelacorte/fieldVMC/commit/338029baa3f934042ad289678c9b9fb3b892543d))
* **deps:** update node.js to 22.14 ([#155](https://github.com/angelacorte/fieldVMC/issues/155)) ([0c9739c](https://github.com/angelacorte/fieldVMC/commit/0c9739ca1bf91e33685ad7f2a4947727b6d70f61))
* **deps:** update python docker tag to v3.13.2 ([b84765d](https://github.com/angelacorte/fieldVMC/commit/b84765dd279eaaa818b3333997120f4b96361e16))

### Bug Fixes

* missing class name refactor ([c9c4d39](https://github.com/angelacorte/fieldVMC/commit/c9c4d395ad96cc26aa711753a58baeb77ffcf682))
* reset steps checked counter ([5e1ceb5](https://github.com/angelacorte/fieldVMC/commit/5e1ceb5e388c8d51b960a1431b9e2c25ab4e06c6))

### Build and continuous integration

* add metrics termination condition for batch mode ([8d7440d](https://github.com/angelacorte/fieldVMC/commit/8d7440d0af10e32d50e7f0539a68ac1f48658e81))
* **deps:** update actions/setup-python action to v5.4.0 ([bf47920](https://github.com/angelacorte/fieldVMC/commit/bf47920767870f7def9eb6ecaa58f54544ee9d34))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.9 ([c25603e](https://github.com/angelacorte/fieldVMC/commit/c25603e7f5f08fa42b04a86099edc4abff85e33a))
* remove protelis dependency ([188d259](https://github.com/angelacorte/fieldVMC/commit/188d259f6d9f253582b3b4ea78c326d49e2203d7))

### General maintenance

* **release:** update gradle.properties .env versions to 1.9.0 [skip ci] ([7101430](https://github.com/angelacorte/fieldVMC/commit/7101430e15d86e8f66643fb629769fc569613c17))

### Style improvements

* use when instead of ifs ([da3d0f6](https://github.com/angelacorte/fieldVMC/commit/da3d0f6a5cb21456ed67649a5850e39040aa5831))

### Refactoring

* remove useless launcher definition ([556d641](https://github.com/angelacorte/fieldVMC/commit/556d641b7344280e0f6d4c8c319b423945f6f659))

## [1.9.0](https://github.com/angelacorte/fieldVMC/compare/1.8.7...1.9.0) (2025-02-18)

### Features

* add also termination time ([5b5f943](https://github.com/angelacorte/fieldVMC/commit/5b5f9432aa0cc8a552acbe3960d02777336d4b77))
* add data retriever object ([aed47e3](https://github.com/angelacorte/fieldVMC/commit/aed47e3c6184084b8c1cb0b8f6d512ff50e4c204))
* add function to evaluate the mean on cleaned data ([f8fd08a](https://github.com/angelacorte/fieldVMC/commit/f8fd08a54ae04e71ea5b62c1e9d593960091382f))
* add new extractors to simulation YAMLs ([40ef94a](https://github.com/angelacorte/fieldVMC/commit/40ef94ad09d040aef39af91c556032ee8fa7015f))
* add objective function as variable ([81824a5](https://github.com/angelacorte/fieldVMC/commit/81824a5084a02b51d2126237b6960fc0c51a7e1f))
* add requirement ([c9780cd](https://github.com/angelacorte/fieldVMC/commit/c9780cd79b126f1b929e8a4b9cc88d6036ba1ccd))
* change launcher into NelderMeadLauncher ([981b9dc](https://github.com/angelacorte/fieldVMC/commit/981b9dceb320c4ab0409ddec9d7a6e60dbcc9762))
* create launcher that uses the nelder mead algorithm for parameters optimization ([aef3e98](https://github.com/angelacorte/fieldVMC/commit/aef3e98115bcf42915f3f7275d4d556bb8bb7251))
* create network density extractor ([2e693d5](https://github.com/angelacorte/fieldVMC/commit/2e693d5eb186b9dfbee74d9fe557a08158f50159))
* create network hub extractor ([4150768](https://github.com/angelacorte/fieldVMC/commit/4150768cf1d59f8ad7b2e85050278ca2a8531140))
* create objective class to evaluate environment's properties ([f590913](https://github.com/angelacorte/fieldVMC/commit/f59091391ae2d08dc351c9905bb44d36f59f5437))
* final way to create the symplex ([f354637](https://github.com/angelacorte/fieldVMC/commit/f35463774e52729e579812501b94bad44d5c2c36))
* implement network density by using abstract aggregating double exporter ([f690827](https://github.com/angelacorte/fieldVMC/commit/f69082758f5c79fab820fcd621ba0b2c9028aa83))
* implement parallelism (maybe) ([f5db7ad](https://github.com/angelacorte/fieldVMC/commit/f5db7ad1814a3278b17f1df5bd6ba6fa3c146b11))
* implementation of the nelder mead algorithm ([5d5cf65](https://github.com/angelacorte/fieldVMC/commit/5d5cf65a4ca9f6aa82c5f95ebf942c361f987742))
* nelder mead method works on parallelism set as 1 ([bc1678c](https://github.com/angelacorte/fieldVMC/commit/bc1678ce237c9bcfa60b95fde84939f63d7702d7))
* update launcher parameters ([249f06d](https://github.com/angelacorte/fieldVMC/commit/249f06d645af013a1a34231b699b13928b0fbfae))
* use alchemist's network diameter function ([5e9e60d](https://github.com/angelacorte/fieldVMC/commit/5e9e60d27b01ab27dcafa875313a446ac4f4e654))
* write result on csv ([12f3409](https://github.com/angelacorte/fieldVMC/commit/12f3409ea108c7e81a3b0c192586474367842932))

### Dependency updates

* **deps:** add rrmxmx-kt dependency ([f2ebcda](https://github.com/angelacorte/fieldVMC/commit/f2ebcdac7067bc9a7b878bdfa1412d59fe53273c))
* **deps:** update alchemist to v37.1.8 ([#144](https://github.com/angelacorte/fieldVMC/issues/144)) ([66f1ba3](https://github.com/angelacorte/fieldVMC/commit/66f1ba3dc64cceb94f6ab5f4a7b9fb3cc311362c))
* **deps:** update alchemist to v37.2.0 ([#146](https://github.com/angelacorte/fieldVMC/issues/146)) ([b6f18c7](https://github.com/angelacorte/fieldVMC/commit/b6f18c70ee2e46a2f5d439faa00ff98f762c162b))
* **deps:** update alchemist to v37.2.1 ([#147](https://github.com/angelacorte/fieldVMC/issues/147)) ([ae67fa1](https://github.com/angelacorte/fieldVMC/commit/ae67fa1104d6ef1b10560682fd1f10da04a86453))
* **deps:** update alchemist to v38 ([#152](https://github.com/angelacorte/fieldVMC/issues/152)) ([c4448cf](https://github.com/angelacorte/fieldVMC/commit/c4448cf0a8cc183ed469c4600a3a2a3dc61d8b92))
* **deps:** update alchemist version ([4013ae7](https://github.com/angelacorte/fieldVMC/commit/4013ae748b6fad135c11441ee86057a5d3e0e833))
* **deps:** update dependency it.unibo.alchemist:alchemist-ui-tooling to v37.3.0 ([#148](https://github.com/angelacorte/fieldVMC/issues/148)) ([065f427](https://github.com/angelacorte/fieldVMC/commit/065f427e4210cfc754ed008800b6025d3f45ea4e))

### Bug Fixes

* add missing parameters in extractor ([cf28f6b](https://github.com/angelacorte/fieldVMC/commit/cf28f6b2901da9bcb924d10dc850dde3a3edff8c))
* change dependencies in order to execute field vmc after the classic version ([2c40e48](https://github.com/angelacorte/fieldVMC/commit/2c40e4808cd96aa485998c56860fbe4a6984a068))
* column name ([70e6fb8](https://github.com/angelacorte/fieldVMC/commit/70e6fb85729b6553b50dad79b408b56214a0bf31))
* comment launchers because they are overridden in the build gradle tasks ([f1b9ffd](https://github.com/angelacorte/fieldVMC/commit/f1b9ffd3f22bd2c7c7c6141e75a9a8af259406a8))
* constructor vals order and name ([5d112df](https://github.com/angelacorte/fieldVMC/commit/5d112df00f49e8803ad0f7f763b0747327bde36b))
* divide hub result into coordX and coordY ([3b98f06](https://github.com/angelacorte/fieldVMC/commit/3b98f0681897d9247ff16fb08d2289d71b7e116d))
* immutable simplex ([18d7893](https://github.com/angelacorte/fieldVMC/commit/18d7893d7d2f94271ebb5bfc0522a26a4e8319f5))
* implement by using Alchemist's new `AbstractAggregatingDoubleExporter` ([44d6d49](https://github.com/angelacorte/fieldVMC/commit/44d6d49cf53e26c374e264ddf2917b3c19247c2d))
* max children not on batch ([b5f20f8](https://github.com/angelacorte/fieldVMC/commit/b5f20f8fb0e41ed04df5b77aad7f6aeaffcb13f6))
* metrics to plot ([e38b54d](https://github.com/angelacorte/fieldVMC/commit/e38b54dd1d476a6e354268320a6e4322a21c2adb))
* missing fix on separator ([fe7f56e](https://github.com/angelacorte/fieldVMC/commit/fe7f56e4653a274ef6f517be86e36455b661e618))
* OS-related path separator ([c57b2a0](https://github.com/angelacorte/fieldVMC/commit/c57b2a06bac9ee58a8600c03b63885c3370701cb))
* pass parameters to extractor ([c133b6c](https://github.com/angelacorte/fieldVMC/commit/c133b6c19f3237d02e26e40b945df0d53ecbb21e))
* remove conflicting launcher ([270646a](https://github.com/angelacorte/fieldVMC/commit/270646ac37f43881f25196b0c82c75d1c436b9c5))
* remove non-seed variables ([6a73244](https://github.com/angelacorte/fieldVMC/commit/6a732444ff6068f8c1a17314d5a79531f66b6ca7))
* remove useless function override ([dbe0a8a](https://github.com/angelacorte/fieldVMC/commit/dbe0a8acd1b21611bd0f270b8b6f1fa49b63dea6))
* translated node points into first quarter ([587bc63](https://github.com/angelacorte/fieldVMC/commit/587bc63a17f4fc4a09db8d426d22c9aadcd9937d))
* translated point into first quarter ([8010e7c](https://github.com/angelacorte/fieldVMC/commit/8010e7c400178729b1e7bd1f1085c670a00ad9e1))
* try retrieve data if not present before throwing an error ([4c31f2d](https://github.com/angelacorte/fieldVMC/commit/4c31f2de6b2d8c0c93fa08697cca6e395e5e4d67))
* typo in val name ([f0f9050](https://github.com/angelacorte/fieldVMC/commit/f0f905063f2c1aacb6e0f33a7f3e49164e5454ca))
* update data to show according to new extractors ([1f512f6](https://github.com/angelacorte/fieldVMC/commit/1f512f63dc65dfc41487ef10f9a28315a8a29654))
* update values for network density evaluation ([f73ad7d](https://github.com/angelacorte/fieldVMC/commit/f73ad7d2ce0434e21cd63f0654a4b9e0504baee6))
* use target as punctual value ([d5904cb](https://github.com/angelacorte/fieldVMC/commit/d5904cb45e79ea0fd219d24f21c5eb3146ec92cc))
* window path backslash ([1707c49](https://github.com/angelacorte/fieldVMC/commit/1707c49759f74153801ac3ad7f45fb4a78f40460))
* wrong variable name ([49293c4](https://github.com/angelacorte/fieldVMC/commit/49293c443c8674f645e05de9fd0665b041a85b09))
* wrong variable name ([36fc206](https://github.com/angelacorte/fieldVMC/commit/36fc206da097bbd50d7e2b3f9e048de7b52a2c35))

### Documentation

* add missing documentation ([f8645ed](https://github.com/angelacorte/fieldVMC/commit/f8645ed5a20269a3b6aaa624bf980fd3724a4fe7))
* add missing documentation ([16a320a](https://github.com/angelacorte/fieldVMC/commit/16a320abd89f712748d2a5aeab0e706e535ea118))

### Build and continuous integration

* add task to run the experiments with the nelder mead launcher ([709da64](https://github.com/angelacorte/fieldVMC/commit/709da648a80cd44d07a3bdf1c7066ef8499d494a))

### General maintenance

* add data ([8038b17](https://github.com/angelacorte/fieldVMC/commit/8038b179273e5468ede04458e36ddea671ffd7e6))
* add data ([f57cf45](https://github.com/angelacorte/fieldVMC/commit/f57cf455eed2e975f4be915db6114991fad36df1))
* add placeholder data for charts evaluation ([920c549](https://github.com/angelacorte/fieldVMC/commit/920c549260bffb09fc5e08d3782f0156d7c574b8))
* add todo ([f74a02d](https://github.com/angelacorte/fieldVMC/commit/f74a02d134ffb0cffe7e59f155a6b94d4021a9f7))
* change seed values ([b803c3b](https://github.com/angelacorte/fieldVMC/commit/b803c3b6710b66a5d0f6a1b09e515a131f98932f))
* **release:** update gradle.properties .env versions to 1.8.7 [skip ci] ([a165837](https://github.com/angelacorte/fieldVMC/commit/a165837fc11dead10a7b0f600d7e5e7cd6934aa8))
* remove comments ([4d2dba4](https://github.com/angelacorte/fieldVMC/commit/4d2dba424b2e557a670e3ccadb4a901d9d9bc0fa))
* remove duplicate data ([76f1fad](https://github.com/angelacorte/fieldVMC/commit/76f1fad23f2e5c20a8065b0c9092f1adb3a81c77))
* remove overriding batch command ([1b46d9e](https://github.com/angelacorte/fieldVMC/commit/1b46d9e244b65c833b7f30ba7eef40cabed79b93))
* remove print ([bd24dbd](https://github.com/angelacorte/fieldVMC/commit/bd24dbd02ba996f6a6283b54380bfa6bf501738f))
* remove useless comments ([7d6154d](https://github.com/angelacorte/fieldVMC/commit/7d6154d9acfc671da34e2a8854468883f5465395))
* replace data ([755acea](https://github.com/angelacorte/fieldVMC/commit/755acea41e0c3dcf3b1c27ad96433e319a1ffaf2))
* replace data ([d69c2c3](https://github.com/angelacorte/fieldVMC/commit/d69c2c390eb0372d8d0954068f080bf6f196f8fc))
* replace data ([2d7a4e6](https://github.com/angelacorte/fieldVMC/commit/2d7a4e6dc1bcdf917dd8cd79f03a42bbf240b35a))
* update cleaned data ([044820d](https://github.com/angelacorte/fieldVMC/commit/044820d47615aedad82689bd77dab6f752387f4b))

### Style improvements

* fix according to ktlint ([f07ce87](https://github.com/angelacorte/fieldVMC/commit/f07ce8754bafdcaa427c9c94b892e29d2315e5d7))
* format according to ktlint ([12fb154](https://github.com/angelacorte/fieldVMC/commit/12fb1545bfaed37f9c7365a2e7e35cce550a2296))
* run ktlint formatter ([147159f](https://github.com/angelacorte/fieldVMC/commit/147159f152fc900361843b7d8e756726eb175324))
* run ktlint formatter ([c9fb909](https://github.com/angelacorte/fieldVMC/commit/c9fb90932b3e368eb8d6343ae1b9925446153de5))

### Refactoring

* changing logic (not completed) ([b9e0c93](https://github.com/angelacorte/fieldVMC/commit/b9e0c93ebe761d2edf7f270e408241de1bd70d3e))
* rename target function and add docs ([fbeccdb](https://github.com/angelacorte/fieldVMC/commit/fbeccdb0eaafc93f4463a282e4e9748dab303ec5))
* take simplex as class parameter ([fe8237d](https://github.com/angelacorte/fieldVMC/commit/fe8237debd028fd34cce18a7b0a47bff615321eb))

## [1.8.7](https://github.com/angelacorte/fieldVMC/compare/1.8.6...1.8.7) (2025-02-03)

### Dependency updates

* **core-deps:** update collektive to v13 ([5eedb9e](https://github.com/angelacorte/fieldVMC/commit/5eedb9e5319b8313b6059035db59732e9d43ca4c))
* **deps:** update alchemist to v37.1.7 ([aa5f624](https://github.com/angelacorte/fieldVMC/commit/aa5f62484a2a6d705ad11b07d4e76ce213296c3d))
* **deps:** update dependency gradle to v8.12.1 ([ed564ec](https://github.com/angelacorte/fieldVMC/commit/ed564ec7e601105d075e183f9059200f1fe07ff2))
* **deps:** update dependency xarray to v2025.1.2 ([7ae8ec9](https://github.com/angelacorte/fieldVMC/commit/7ae8ec9ef726b1608fffad47404d5cd9d970801a))
* **deps:** update plugin gitsemver to v4.0.2 ([c17c587](https://github.com/angelacorte/fieldVMC/commit/c17c5872989be42ec0869d183b7328b46d8cc92f))
* **deps:** update plugin kotlin-qa to v0.78.0 ([5000e19](https://github.com/angelacorte/fieldVMC/commit/5000e19b83d74dd79a4d11516426c026388cbf48))
* **deps:** update plugin multijvmtesting to v3.2.2 ([34e343d](https://github.com/angelacorte/fieldVMC/commit/34e343d9753c431f66479147da550b6a249eee8d))

### Bug Fixes

* correct distanceTo import as for Collektive v13 ([844830b](https://github.com/angelacorte/fieldVMC/commit/844830b99277514dd9e89f5981643a15f7c0f9c7))

### General maintenance

* **release:** update gradle.properties .env versions to 1.8.6 [skip ci] ([507aac4](https://github.com/angelacorte/fieldVMC/commit/507aac41d4ef718b3813e9268e680c8879f62d11))

## [1.8.6](https://github.com/angelacorte/fieldVMC/compare/1.8.5...1.8.6) (2025-01-31)

### Dependency updates

* **core-deps:** update collektive to v12.3.3 ([e35ca62](https://github.com/angelacorte/fieldVMC/commit/e35ca62e79b284a9c2fb33e4e259ac933871a6ae))
* **deps:** update alchemist to v37.1.6 ([8b90530](https://github.com/angelacorte/fieldVMC/commit/8b9053091da30d051006dd6d4216aca6a70628f9))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.8 ([f0e29ee](https://github.com/angelacorte/fieldVMC/commit/f0e29ee421a4c2904283a1844018cf37f1c04b23))

### General maintenance

* **release:** update gradle.properties .env versions to 1.8.5 [skip ci] ([53089b7](https://github.com/angelacorte/fieldVMC/commit/53089b7826c28a51db5bec0099578b9bf2f52c37))

## [1.8.5](https://github.com/angelacorte/fieldVMC/compare/1.8.4...1.8.5) (2025-01-27)

### Dependency updates

* **core-deps:** update dependency org.jetbrains.kotlin.jvm to v2.1.10 ([057b0a0](https://github.com/angelacorte/fieldVMC/commit/057b0a042de7ba86c4ea4bdb5ced3e225a44a8cc))
* **deps:** update plugin gitsemver to v3.1.9 ([a4f68af](https://github.com/angelacorte/fieldVMC/commit/a4f68af4321b0f632a334f38fd851b6d568ba31b))
* **deps:** update plugin gitsemver to v4 ([0082ac1](https://github.com/angelacorte/fieldVMC/commit/0082ac1c789649d899add74f07c6beae44747782))
* **deps:** update plugin multijvmtesting to v3.2.1 ([531e8d2](https://github.com/angelacorte/fieldVMC/commit/531e8d2538026a29acf99becaa38aa5f6a5ee312))

### Build and continuous integration

* **deps:** update actions/setup-node action to v4.2.0 ([1c9b91c](https://github.com/angelacorte/fieldVMC/commit/1c9b91c5b0eecb05952f4b8b00d12400a4ca05a1))

### General maintenance

* **release:** update gradle.properties .env versions to 1.8.4 [skip ci] ([9b3d5a3](https://github.com/angelacorte/fieldVMC/commit/9b3d5a325769ab8daadd242049ad2bcb8b9c0545))

## [1.8.4](https://github.com/angelacorte/fieldVMC/compare/1.8.3...1.8.4) (2025-01-26)

### Dependency updates

* **core-deps:** update collektive to v12.3.1 ([3cda2a0](https://github.com/angelacorte/fieldVMC/commit/3cda2a00cbe90456c03049643ac4796ec917e15c))
* **deps:** update alchemist to v37.1.3 ([99ccd65](https://github.com/angelacorte/fieldVMC/commit/99ccd658d4af664452605781fc2b31b770dbc815))
* **deps:** update alchemist to v37.1.4 ([78b9779](https://github.com/angelacorte/fieldVMC/commit/78b97795ab6b7e6a9e120677ed308de48da6d5aa))
* **deps:** update alchemist to v37.1.5 ([35a3211](https://github.com/angelacorte/fieldVMC/commit/35a3211889102d568af172d5c7e92074333a365e))
* **deps:** update plugin com.gradle.enterprise to v3.19.1 ([aedea3a](https://github.com/angelacorte/fieldVMC/commit/aedea3acb3b73a1e76a771aa29292cc337027971))
* **deps:** update plugin gitsemver to v3.1.8 ([c31c800](https://github.com/angelacorte/fieldVMC/commit/c31c800049704ef0f27f4e15fbbe851135abf359))
* **deps:** update plugin multijvmtesting to v3.2.0 ([6bb4fa0](https://github.com/angelacorte/fieldVMC/commit/6bb4fa0e2be31b9880a5bf1fd4e025ee06f99fa3))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.3 ([f08f844](https://github.com/angelacorte/fieldVMC/commit/f08f8447d655b18f5568a6d49615d40a62ef8390))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.6 ([dab758b](https://github.com/angelacorte/fieldVMC/commit/dab758bc38fc77cccfef9ac9219e028572f8c015))

### General maintenance

* **release:** update gradle.properties .env versions to 1.8.3 [skip ci] ([50d0efe](https://github.com/angelacorte/fieldVMC/commit/50d0efe898d59a135458f83df8ca2576449d19b1))

## [1.8.3](https://github.com/angelacorte/fieldVMC/compare/1.8.2...1.8.3) (2025-01-21)

### Bug Fixes

* change way to evaluate network degree in classicVMC ([af8b198](https://github.com/angelacorte/fieldVMC/commit/af8b1983949c9a2ed6d9e2ee5e49e2130c52dd48))

### General maintenance

* minors in simulations setup ([7453a24](https://github.com/angelacorte/fieldVMC/commit/7453a24a4c27fd9a7173ca8c4b24dd9fe9b42a74))
* **release:** update gradle.properties .env versions to 1.8.2 [skip ci] ([7434a02](https://github.com/angelacorte/fieldVMC/commit/7434a028e96239154e047de83297e8c3309e76d2))

### Refactoring

* correct name of metric node degree ([2d880af](https://github.com/angelacorte/fieldVMC/commit/2d880af2f8b3cf4b511b251edb4ceb6bc814704a))

## [1.8.2](https://github.com/angelacorte/fieldVMC/compare/1.8.1...1.8.2) (2025-01-21)

### Dependency updates

* **core-deps:** update collektive to v12.3.0 ([#118](https://github.com/angelacorte/fieldVMC/issues/118)) ([5cf56e0](https://github.com/angelacorte/fieldVMC/commit/5cf56e023f86e5020e42b04fb11bafeb418f7063))

### General maintenance

* **release:** update gradle.properties .env versions to 1.8.1 [skip ci] ([f7e34bb](https://github.com/angelacorte/fieldVMC/commit/f7e34bbc4004b33c39a82fd8473e758f9bee4078))

## [1.8.1](https://github.com/angelacorte/fieldVMC/compare/1.8.0...1.8.1) (2025-01-21)

### Dependency updates

* **deps:** update alchemist to v37 ([cb38b2d](https://github.com/angelacorte/fieldVMC/commit/cb38b2d6e5a71a3d4a3edf5f7134a35ce9f2fb15))

### Bug Fixes

* adapt to new alchemist version ([0c612ac](https://github.com/angelacorte/fieldVMC/commit/0c612ac1b1df782e9862e0bf87c9ef2cd9b4c8f9))

### General maintenance

* **release:** update gradle.properties .env versions to 1.8.0 [skip ci] ([f62ea39](https://github.com/angelacorte/fieldVMC/commit/f62ea39f4a7e5be4e1c6b7f0ee06c258e22536a5))
* restore placeholder data ([de18617](https://github.com/angelacorte/fieldVMC/commit/de1861763654758bf1a887350b47b4723aac84f0))

## [1.8.0](https://github.com/angelacorte/fieldVMC/compare/1.7.1...1.8.0) (2025-01-20)

### Features

* remove the wait before spawning ([2aa2e7f](https://github.com/angelacorte/fieldVMC/commit/2aa2e7fb2667afa70f5e8cc49dc731b7251f78af))

### Bug Fixes

* adapt simulations to be similar ([9d50cf8](https://github.com/angelacorte/fieldVMC/commit/9d50cf835258783975729aa286fd6ed65f0875f0))

### General maintenance

* add placeholder data ([40c9746](https://github.com/angelacorte/fieldVMC/commit/40c9746379d4f1ef8a4b46e8df2d71625d12c81a))
* delete old data ([17607ac](https://github.com/angelacorte/fieldVMC/commit/17607ac3998c383099fbdbab88100f391155445c))
* **release:** update gradle.properties .env versions to 1.7.1 [skip ci] ([bd6ed86](https://github.com/angelacorte/fieldVMC/commit/bd6ed86a7238fc0b62efbf8f4279b1bbd625c814))
* remove useless aggregator min from children-count ([0118190](https://github.com/angelacorte/fieldVMC/commit/0118190c99968553a6d950f361edca093e47b9e8))

## [1.7.1](https://github.com/angelacorte/fieldVMC/compare/1.7.0...1.7.1) (2025-01-20)

### Dependency updates

* **core-deps:** update collektive to v12.2.0 ([7c92739](https://github.com/angelacorte/fieldVMC/commit/7c9273993c45e4d9317244e5a8915d94fdd964c8))

### Bug Fixes

* **ci:** change simulation check in CI ([8125b96](https://github.com/angelacorte/fieldVMC/commit/8125b96a1c8ec2b2392bd7e07613484497c0cd74))
* **ci:** change simulation check in CI ([80851dc](https://github.com/angelacorte/fieldVMC/commit/80851dcdb504b311a387f497a418b8b08b7b5d4b))
* remove competition rate from batch ([adb9118](https://github.com/angelacorte/fieldVMC/commit/adb91188f840dda3252f4e2720849f47dadf0399))
* remove unused env variables in dockerfile ([2f57f7f](https://github.com/angelacorte/fieldVMC/commit/2f57f7f9eefe4c43db91df1ff0c3ae61b0e03303))

### General maintenance

* adapt classic vmc batch values to fieldvmc ([9d57cac](https://github.com/angelacorte/fieldVMC/commit/9d57caca64eee583213027f8a0830e5b6a31605f))
* **build:** override launcher parameters in ci ([8651950](https://github.com/angelacorte/fieldVMC/commit/8651950f4ef2c6916104d185af1129d55519f4ed))
* **release:** update gradle.properties .env versions to 1.7.0 [skip ci] ([7227b43](https://github.com/angelacorte/fieldVMC/commit/7227b43131856324610c2e2632ab064a75a3b230))

### Style improvements

* run ktlint formatter ([b6265f9](https://github.com/angelacorte/fieldVMC/commit/b6265f94852dd0c3a0213ca4efd91903f487f8c1))

## [1.7.0](https://github.com/angelacorte/fieldVMC/compare/1.6.0...1.7.0) (2025-01-17)

### Features

* add launcher specs for batch ([0c687c6](https://github.com/angelacorte/fieldVMC/commit/0c687c63ce62cc260ce56a40dad54f1b7b3eec7b))
* improve plotting with stdev area ([01e68a9](https://github.com/angelacorte/fieldVMC/commit/01e68a9c97d2c21d6c61cc63fce74fc4fc6cfab2))

### General maintenance

* change simulation default parameters to be as similar as possible ([96d85a5](https://github.com/angelacorte/fieldVMC/commit/96d85a536a60cd85b9c073811aad5ce1e395a8b5))
* **release:** update gradle.properties .env versions to 1.6.0 [skip ci] ([e88fa8b](https://github.com/angelacorte/fieldVMC/commit/e88fa8b775ab1258b86cbc16910292d1f10f7220))

## [1.6.0](https://github.com/angelacorte/fieldVMC/compare/1.5.2...1.6.0) (2025-01-16)

### Features

* add new extractor in classic vmc ([7275a96](https://github.com/angelacorte/fieldVMC/commit/7275a96107a5d138ec07ba7b7338ad058acfe24b))

### Build and continuous integration

* **deps:** update actions/upload-artifact action to v4.6.0 ([83dd3cb](https://github.com/angelacorte/fieldVMC/commit/83dd3cb4d7a60851b95dac8db8b8349028378596))

### General maintenance

* delete old data ([d4b2c1d](https://github.com/angelacorte/fieldVMC/commit/d4b2c1ddac3f439e2dd29d1cbd2e285c50041d81))
* **release:** update gradle.properties .env versions to 1.5.2 [skip ci] ([ce1c365](https://github.com/angelacorte/fieldVMC/commit/ce1c36528a88ff9c74c7d10a07553899e475b5a4))

### Refactoring

* change plotters (work in progress) ([a012302](https://github.com/angelacorte/fieldVMC/commit/a012302842af3cc2f1835a4d1735b332e2a23df0))

## [1.5.2](https://github.com/angelacorte/fieldVMC/compare/1.5.1...1.5.2) (2025-01-16)

### Dependency updates

* **core-deps:** update collektive to v12.1.1 ([799a98d](https://github.com/angelacorte/fieldVMC/commit/799a98dd53973dffffdea2e22aec516fc61dbacb))
* **deps:** update node.js to 22.13 ([c13de1a](https://github.com/angelacorte/fieldVMC/commit/c13de1ae994f90a61cf14eb5f2ac03972e15fbed))
* **deps:** update plugin multijvmtesting to v3 ([d6bac85](https://github.com/angelacorte/fieldVMC/commit/d6bac853bfe72e071facc4f73f1f55a7394a37be))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.7.0 ([ce857f7](https://github.com/angelacorte/fieldVMC/commit/ce857f7b28230c4331bab095edf9bfd9044944ed))

### General maintenance

* **release:** update gradle.properties .env versions to 1.5.1 [skip ci] ([bb7baf0](https://github.com/angelacorte/fieldVMC/commit/bb7baf0212f2417eba33783ceba1ecff9040203e))

## [1.5.1](https://github.com/angelacorte/fieldVMC/compare/1.5.0...1.5.1) (2025-01-16)

### Dependency updates

* **core-deps:** update collektive ([029273c](https://github.com/angelacorte/fieldVMC/commit/029273c5fb7f76cf2be3a2bb1a517877ce918c05))
* **core-deps:** update dependency org.jetbrains.kotlin.jvm to v2.1.0 ([51789b3](https://github.com/angelacorte/fieldVMC/commit/51789b3a887b812862ad022fbde60dfe139778fd))
* **deps:** update alchemist to v36.1.3 ([228eb02](https://github.com/angelacorte/fieldVMC/commit/228eb029440c69c01a0b043a2cc09ae862f3e231))
* **deps:** update dependency xarray to v2025.1.1 ([30544e4](https://github.com/angelacorte/fieldVMC/commit/30544e4ba4b6d75ee03016853fe70cd06f47153b))

### Bug Fixes

* restore seeds ([982da49](https://github.com/angelacorte/fieldVMC/commit/982da4982793de8bf442148ea7b7c3347f5da5fc))

### General maintenance

* **release:** update gradle.properties .env versions to 1.5.0 [skip ci] ([7dab028](https://github.com/angelacorte/fieldVMC/commit/7dab02885748869c1394450a8fe81e359784987c))

## [1.5.0](https://github.com/angelacorte/fieldVMC/compare/1.4.9...1.5.0) (2025-01-16)

### Features

* add custom extractors ([c03dfc6](https://github.com/angelacorte/fieldVMC/commit/c03dfc641e89f9259480810b31246723b7946a91))
* add extractors to simulation yaml ([88278b0](https://github.com/angelacorte/fieldVMC/commit/88278b0a735d0429745de4513c489f0744676386))
* add network degree extractor ([1b0e14e](https://github.com/angelacorte/fieldVMC/commit/1b0e14e54307ce2b928c7a3a018adaec8d1fad55))
* add network diameter extractor ([3dbf068](https://github.com/angelacorte/fieldVMC/commit/3dbf0682b475b66348dfd35c49914d5bdc7d6bf7))
* add new python plotter ([dd742d2](https://github.com/angelacorte/fieldVMC/commit/dd742d2fe04f47ec081dec2414d9f05cb11f0808))
* add termination for batch experiments ([31c7bb1](https://github.com/angelacorte/fieldVMC/commit/31c7bb1fe360110392823f95cae1226c06a84385))
* change simulations termination type ([0ee4e06](https://github.com/angelacorte/fieldVMC/commit/0ee4e065d8f45e2269d1a3f3af8d4e711566a2a8))
* dockerize classicVMC and fixed leader FieldVMC simulations ([94715b6](https://github.com/angelacorte/fieldVMC/commit/94715b6360af21b293fdee99b2f9fd272fed1d74))

### Dependency updates

* **deps:** update alpine docker tag to v3.21.2 ([b112ed5](https://github.com/angelacorte/fieldVMC/commit/b112ed5b8c6cc31c84223f17078fd3df34c78cb7))

### Bug Fixes

* add seeds ([5e63e9f](https://github.com/angelacorte/fieldVMC/commit/5e63e9f49cdd26d9bc1ee5eb9629b2cc440ef3b2))
* add variable for spawning wait ([02261ff](https://github.com/angelacorte/fieldVMC/commit/02261ff1e8eda9c2e8f53d3ddf2611268690260c))
* **ci:** simulation ARG ([a636503](https://github.com/angelacorte/fieldVMC/commit/a6365034ff96c52281f26fea74f700f236cd9b11))
* comment create charts in build adn deploy ([8514325](https://github.com/angelacorte/fieldVMC/commit/8514325a37b0bb5ef0018054dba300aa92c680db))
* comment create charts in build adn deploy ([d0c3b68](https://github.com/angelacorte/fieldVMC/commit/d0c3b68d7078b00e17bb1380cdd837cafad3ee98))
* comment create charts in build adn deploy ([627de10](https://github.com/angelacorte/fieldVMC/commit/627de100f5f21156503c95ee8a8f71f524472bdb))
* remove classic-vmc from data to evaluate ([236488f](https://github.com/angelacorte/fieldVMC/commit/236488f5281d53c09fdfd1cf41efa5e444070e1a))
* remove destruction of nodes in fieldVMC ([ee2d210](https://github.com/angelacorte/fieldVMC/commit/ee2d21048b000d2e759bb4f33631a75d3817317a))
* remove useless ARGs ([b6ce804](https://github.com/angelacorte/fieldVMC/commit/b6ce8040edc6bf08a280089f56131ea4927a5e6e))
* remove useless environment variables ([073f01c](https://github.com/angelacorte/fieldVMC/commit/073f01c5529689bcd99be11b567e08ce52ca8cb9))
* rollback to charts evaluation ([506a3e0](https://github.com/angelacorte/fieldVMC/commit/506a3e0145991e459a442eb870dd03747c6d1916))
* temporary disable charts generation in docker compose ([fb549d9](https://github.com/angelacorte/fieldVMC/commit/fb549d98b53f7b93c1bfdf4c39f82a9741eca868))
* temporary disable successful condition on charts generation ([39b28e2](https://github.com/angelacorte/fieldVMC/commit/39b28e23c8dea255d14b1ae3f26ee6ef46ba1a9e))
* temporary remove charts from release ([e626f61](https://github.com/angelacorte/fieldVMC/commit/e626f6138d1068b570fd2463577928b604ed8ea7))
* update ENV variables and ARG ([9b875c5](https://github.com/angelacorte/fieldVMC/commit/9b875c5dbc76ef573c13c260b80e1ddb17cd2fe6))
* wrong dependency for charts ([dbba171](https://github.com/angelacorte/fieldVMC/commit/dbba171cf7352a8757d2e820eadb9c1e359e8d36))
* wrong Dockerfile location ([5e294b2](https://github.com/angelacorte/fieldVMC/commit/5e294b28fde80560d96e49204f0727f5847f0810))

### General maintenance

* add children count molecule ([d19c755](https://github.com/angelacorte/fieldVMC/commit/d19c7557301dd0b5b649c15b7911e6b98ae54d49))
* **build:** remove override in batch simulations ([ce15345](https://github.com/angelacorte/fieldVMC/commit/ce15345deb5f677a01982ba5b0143d831189b6c8))
* change max seed ([5100380](https://github.com/angelacorte/fieldVMC/commit/51003809aebd48cabeb3e79d38fbd7637448392a))
* **release:** update gradle.properties .env versions to 1.4.9 [skip ci] ([424eb8d](https://github.com/angelacorte/fieldVMC/commit/424eb8d0c9d4c62186f641ac417fe52939bcdbf6))
* remove useless comments ([0d2af35](https://github.com/angelacorte/fieldVMC/commit/0d2af35768620b7536ecb73ccc4ee7afa41e5ee7))

### Style improvements

* put variables in lexicographic order ([e4abf53](https://github.com/angelacorte/fieldVMC/commit/e4abf53b4ac2ffa321e2680d9ec95d2a9dd6fd7c))

## [1.4.9](https://github.com/angelacorte/fieldVMC/compare/1.4.8...1.4.9) (2025-01-07)

### Bug Fixes

* manage angle NaN case ([1e95be9](https://github.com/angelacorte/fieldVMC/commit/1e95be997aacaae44c3a21ed7a60096ba4743ee6))

### General maintenance

* add prompt command to launch simulation ([23de41f](https://github.com/angelacorte/fieldVMC/commit/23de41f16dbb8ecf156f84f23d650b0c67c53ef7))
* **release:** update gradle.properties .env versions to 1.4.8 [skip ci] ([358ecc9](https://github.com/angelacorte/fieldVMC/commit/358ecc9724f6c13fda72fda39908deb470a179d4))

## [1.4.8](https://github.com/angelacorte/fieldVMC/compare/1.4.7...1.4.8) (2025-01-07)

### Dependency updates

* **deps:** update dependency xarray to v2025 ([1baf925](https://github.com/angelacorte/fieldVMC/commit/1baf9251d5532ba980c0428ae9e45bfe526e9ac1))

### Bug Fixes

* transfer success evaluation ([f3f1357](https://github.com/angelacorte/fieldVMC/commit/f3f1357edf370052f8cdc4c83f65fdb3b3dbc7dd))

### General maintenance

* add clarifying comments ([348857c](https://github.com/angelacorte/fieldVMC/commit/348857ccef840bd97f44c26e4e0e276785bcb588))
* change layer position ([d11fddd](https://github.com/angelacorte/fieldVMC/commit/d11fddd5491e6ef1d1a28be20c1f0e1def31c1ab))
* **release:** update gradle.properties .env versions to 1.4.7 [skip ci] ([83cde64](https://github.com/angelacorte/fieldVMC/commit/83cde643d9a93bf2dca6245048c103706b08b78a))

### Refactoring

* change val name ([fd9d641](https://github.com/angelacorte/fieldVMC/commit/fd9d641c7057e5c9e3229407afc0f23c9a69558a))

## [1.4.7](https://github.com/angelacorte/fieldVMC/compare/1.4.6...1.4.7) (2025-01-07)

### Bug Fixes

* add check on spawning ([3c3dee4](https://github.com/angelacorte/fieldVMC/commit/3c3dee471f6d2b16f0d7eab75749cbee5c3fea13))

### General maintenance

* **release:** update gradle.properties .env versions to 1.4.6 [skip ci] ([25e0ed1](https://github.com/angelacorte/fieldVMC/commit/25e0ed14a52665bc2e278c66276320f080c2285a))

## [1.4.6](https://github.com/angelacorte/fieldVMC/compare/1.4.5...1.4.6) (2024-12-28)

### Dependency updates

* **core-deps:** update collektive to v12 ([bf1223e](https://github.com/angelacorte/fieldVMC/commit/bf1223e2fa6469232366caf853df7cb5c8ffd816))
* **deps:** update dependency gradle to v8.12 ([8c37314](https://github.com/angelacorte/fieldVMC/commit/8c3731468be468c9016d8d5620352845d7295d6a))

### General maintenance

* **release:** update gradle.properties .env versions to 1.4.5 [skip ci] ([d62fc64](https://github.com/angelacorte/fieldVMC/commit/d62fc6499f4a0f6cc913bf09c74a17bbd6a6fbb6))

## [1.4.5](https://github.com/angelacorte/fieldVMC/compare/1.4.4...1.4.5) (2024-12-27)

### Dependency updates

* **core-deps:** update collektive to v11.2.0 ([096a623](https://github.com/angelacorte/fieldVMC/commit/096a62314eeb457e63a98e7151d35819f9bf7c8c))
* **deps:** update alchemist to v36.0.12 ([#102](https://github.com/angelacorte/fieldVMC/issues/102)) ([8c0fd3a](https://github.com/angelacorte/fieldVMC/commit/8c0fd3a457da27940914271d6d9dfb14ee0a1d9d))

### General maintenance

* **release:** update gradle.properties .env versions to 1.4.4 [skip ci] ([f55a113](https://github.com/angelacorte/fieldVMC/commit/f55a11345ae30429df7fe5dec41cb204d18b3e3a))

## [1.4.4](https://github.com/angelacorte/fieldVMC/compare/1.4.3...1.4.4) (2024-12-27)

### Bug Fixes

* changing logic (again) using should use MAX as status to understand who should spawn ([029ce33](https://github.com/angelacorte/fieldVMC/commit/029ce33bf2d9c9f0f4258bc951bb4ee4d080d9e1))
* logic to spawn passing through a find max node phase ([14c6404](https://github.com/angelacorte/fieldVMC/commit/14c64049e2e406ec81f8b03f739945f367c48a3a))

### General maintenance

* **release:** update gradle.properties .env versions to 1.4.3 [skip ci] ([1927491](https://github.com/angelacorte/fieldVMC/commit/1927491846d4a6a3342972b3a90fc77fbbed6058))

## [1.4.3](https://github.com/angelacorte/fieldVMC/compare/1.4.2...1.4.3) (2024-12-20)

### Dependency updates

* **core-deps:** update collektive to v11.1.3 ([f9446e7](https://github.com/angelacorte/fieldVMC/commit/f9446e731e20766d27e4769753ab0184cfdca657))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.117 ([1867da6](https://github.com/angelacorte/fieldVMC/commit/1867da6922c4de3f40a5c8a3f9e8ab588f4de012))

### General maintenance

* **release:** update gradle.properties .env versions to 1.4.2 [skip ci] ([b2d119d](https://github.com/angelacorte/fieldVMC/commit/b2d119d635770e3560322bb770f723e32bb10fc8))

## [1.4.2](https://github.com/angelacorte/fieldVMC/compare/1.4.1...1.4.2) (2024-12-19)

### Dependency updates

* **deps:** update alchemist to v36.0.10 ([c1d684c](https://github.com/angelacorte/fieldVMC/commit/c1d684cde529cf35614afc1e164d0a0c557fd335))
* **deps:** update alchemist to v36.0.11 ([d1ccc8c](https://github.com/angelacorte/fieldVMC/commit/d1ccc8c79277c52608412a79efd6b125ebfdbce5))

### Bug Fixes

* update success sensing cases ([27e1fb7](https://github.com/angelacorte/fieldVMC/commit/27e1fb7c1002210173af41a4570c029eb11d6f08))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.28 ([12a4170](https://github.com/angelacorte/fieldVMC/commit/12a41708c73cc2fc53ea1ad0f01143feb25e7d5b))

### General maintenance

* **release:** update gradle.properties .env versions to 1.4.1 [skip ci] ([5316310](https://github.com/angelacorte/fieldVMC/commit/531631041de7a216ecf28197fc8c46ffddbe547e))

## [1.4.1](https://github.com/angelacorte/fieldVMC/compare/1.4.0...1.4.1) (2024-12-18)

### Bug Fixes

* nodes in deadlock for success evaluation - still work in progress ([8ced19b](https://github.com/angelacorte/fieldVMC/commit/8ced19bd539139dba0b8a8bbe8a9806656d3d81f))

### General maintenance

* **release:** update gradle.properties .env versions to 1.4.0 [skip ci] ([0d91ea0](https://github.com/angelacorte/fieldVMC/commit/0d91ea02c4e2ea1108f9c84212f770b0c91e77e1))

## [1.4.0](https://github.com/angelacorte/fieldVMC/compare/1.3.0...1.4.0) (2024-12-18)

### Features

* spawn more than one node ([e6ef6c7](https://github.com/angelacorte/fieldVMC/commit/e6ef6c7372cbac7f8664782b38cfe62ea8ae1cee))

### Dependency updates

* **deps:** update alchemist to v36.0.9 ([398b82c](https://github.com/angelacorte/fieldVMC/commit/398b82c7625f22b11cc81229253d74220c0833ac))
* **deps:** update dependency matplotlib to v3.10.0 ([c1af005](https://github.com/angelacorte/fieldVMC/commit/c1af005c2efac4f7dc44f9f4412bca09dc19c0e0))
* **deps:** update dependency matplotlib to v3.9.4 ([a3c695e](https://github.com/angelacorte/fieldVMC/commit/a3c695e5028822f30c2e1084518b7486221632ee))
* **deps:** update node.js to 22.12 ([f6ef210](https://github.com/angelacorte/fieldVMC/commit/f6ef210782223275e03c137b1962aab49c7c02b1))
* **deps:** update plugin com.gradle.enterprise to v3.19 ([a1ade25](https://github.com/angelacorte/fieldVMC/commit/a1ade2517a52f9325df39c81c951ee0810d7414a))
* **deps:** update python docker tag to v3.13.1 ([abda2d1](https://github.com/angelacorte/fieldVMC/commit/abda2d1486c71eb0e3f94de29906c3f25b1e35c7))

### Bug Fixes

* always add resources to root ([3701271](https://github.com/angelacorte/fieldVMC/commit/37012717567cb1ec8c680e3fb4d4ea310e3af9c7))

### Build and continuous integration

* **deps:** update actions/upload-artifact action to v4.5.0 ([f1cc5a5](https://github.com/angelacorte/fieldVMC/commit/f1cc5a5baf055412e01620132ba15f6fc0d64f36))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.25 ([f652bbf](https://github.com/angelacorte/fieldVMC/commit/f652bbf67b4b55d249db6fefe3be96a8165547b2))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.26 ([b9e51ee](https://github.com/angelacorte/fieldVMC/commit/b9e51eeddd2ec5560e6a7618991aa78285554bcc))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.27 ([83085c9](https://github.com/angelacorte/fieldVMC/commit/83085c915f74f19f1543aa06db93fdf41c94ccb2))

### General maintenance

* change position of success layer ([b255e39](https://github.com/angelacorte/fieldVMC/commit/b255e39e5473a7a11412e2e569caac6b1269dfff))
* **release:** update gradle.properties .env versions to 1.3.0 [skip ci] ([b0cfa92](https://github.com/angelacorte/fieldVMC/commit/b0cfa921d117a8acbf7dabe453bac80a42b17326))

### Refactoring

* remove lateinit clock ([68d6039](https://github.com/angelacorte/fieldVMC/commit/68d6039d191aba2d021f3d6c5967cf2e8e1f83e4))

## [1.3.0](https://github.com/angelacorte/fieldVMC/compare/1.2.0...1.3.0) (2024-12-17)

### Features

* add (empty) actions for classic vmc ([40fafbc](https://github.com/angelacorte/fieldVMC/commit/40fafbc0d50768813fea220a0b94ee05b99953d4))
* add `spawned` action in cycle ([6c30f5a](https://github.com/angelacorte/fieldVMC/commit/6c30f5aaa9d6ea5527f1184cd9cbe5165ba0ffd1))
* add base yaml for classic vmc sim ([0e8217f](https://github.com/angelacorte/fieldVMC/commit/0e8217f68269ac21ed18312554eaed6b781dd330))
* add check on execution clock ([ce62958](https://github.com/angelacorte/fieldVMC/commit/ce6295868b2fa28ff26c01f34abbd1ebc4022ae9))
* add check on execution clock ([742af1a](https://github.com/angelacorte/fieldVMC/commit/742af1aa31510b3326ba3e4dcb35a92996de3bb4))
* add class for custom linking rule ([0e207dc](https://github.com/angelacorte/fieldVMC/commit/0e207dcd7a58004b67314a526b481c9aba05fedc))
* add entrypoint class for classic vmc ([bc80c3a](https://github.com/angelacorte/fieldVMC/commit/bc80c3a8b86b5ef6978a6387c8cb775f74cb48c1))
* add execution clock ([1f7b85a](https://github.com/angelacorte/fieldVMC/commit/1f7b85ae67c9766558fab2002d0c7abc7ed355de))
* add first implementation of clock property ([3dd1a54](https://github.com/angelacorte/fieldVMC/commit/3dd1a54b392f744b0bb48ec21034062bfadca3b6))
* add main resource distribution logic ([acd3099](https://github.com/angelacorte/fieldVMC/commit/acd309901d43354467af400dc98447a57be56826))
* add main spawn logic ([896c8b2](https://github.com/angelacorte/fieldVMC/commit/896c8b25effed231aeb1efa9042a56102ca90a49))
* add main success sensing logic ([9c2bdc2](https://github.com/angelacorte/fieldVMC/commit/9c2bdc22f7cc90b04aa5205b50d92a6324a3dfc1))
* add main weight update logic ([bdfae3f](https://github.com/angelacorte/fieldVMC/commit/bdfae3f0474c0a4087b20318d38626b1f3fec87c))
* add parameters ([86a4299](https://github.com/angelacorte/fieldVMC/commit/86a42990f04fec16b87b3844eb17c66a5085dfcc))
* add setup for just spawned nodes ([2720290](https://github.com/angelacorte/fieldVMC/commit/2720290f62bd8803149420f48d94a99fc117aa6b))
* dockerize project ([420a0e6](https://github.com/angelacorte/fieldVMC/commit/420a0e698c5ad8064146fd32e5b17a96373f864a))
* merge weight update and resource distribution ([432abe4](https://github.com/angelacorte/fieldVMC/commit/432abe41d0926c425502570e3e88db290fefc002))
* parametrize simulation and retrieve data ([4a000f1](https://github.com/angelacorte/fieldVMC/commit/4a000f148ac22086d0a2795109e4092c366938f6))

### Dependency updates

* **deps:** update alpine docker tag to v3.21.0 ([#87](https://github.com/angelacorte/fieldVMC/issues/87)) ([ec4e9fd](https://github.com/angelacorte/fieldVMC/commit/ec4e9fdcdac8303a262da194b88dfb910c780ade))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.116 ([ebdf130](https://github.com/angelacorte/fieldVMC/commit/ebdf13021be31d92f583c9edd11125d54baf5624))
* **deps:** update plugin kotlin-qa to v0.78.0 ([#82](https://github.com/angelacorte/fieldVMC/issues/82)) ([63a9221](https://github.com/angelacorte/fieldVMC/commit/63a9221be7439b4f1599af0a26688406045f418c))

### Bug Fixes

* add check on node id ([0ee247a](https://github.com/angelacorte/fieldVMC/commit/0ee247a17c0c6a896131edd6dc823bd93e5f554e))
* add starting resource for leader ([d5a037d](https://github.com/angelacorte/fieldVMC/commit/d5a037df21f1453427108bd2a51da175a5abd6d9))
* change action parameters into node properties ([0874080](https://github.com/angelacorte/fieldVMC/commit/0874080d305db47c0920f57280658985c80e3e6c))
* change task to execute for checking simulations ([ee19902](https://github.com/angelacorte/fieldVMC/commit/ee19902a30e91175347038881bab7f1796fe49d6))
* **ci:** change simulation check ([720243d](https://github.com/angelacorte/fieldVMC/commit/720243d0d98363879cfba0e714068ced12336df3))
* **ci:** typo in docker command ([ce63719](https://github.com/angelacorte/fieldVMC/commit/ce6371978ec3c793e6a1000f3c95eeb96d6bb9be))
* different input for success value ([732b0e6](https://github.com/angelacorte/fieldVMC/commit/732b0e639e9d91633d5b1dacb1f5ef8ed8c9eaef))
* leader concentration into root ([b9b3012](https://github.com/angelacorte/fieldVMC/commit/b9b3012cd5dd6e7536a86bef5f967cb7ae44bf83))
* missing increase clock ([70b54d0](https://github.com/angelacorte/fieldVMC/commit/70b54d0a3706baec9032d68ed749b524c08ae5c6))
* move classic vmc-related constant into action ([1f096b8](https://github.com/angelacorte/fieldVMC/commit/1f096b8dd7723fef83d230b781f164a161cda006))
* parametrized in docker ([26b3d64](https://github.com/angelacorte/fieldVMC/commit/26b3d642a633d9a45d2bbbba7d232b3d607293fa))
* remove update weight action ([9528bef](https://github.com/angelacorte/fieldVMC/commit/9528bef44648c96d66027939c765b6af832457da))
* remove useless call to mutable list ([588b86f](https://github.com/angelacorte/fieldVMC/commit/588b86ff03c16f673d0bdb1c1ca086ee03bfc185))
* remove useless class ([6dd2e90](https://github.com/angelacorte/fieldVMC/commit/6dd2e90f85b86d33b8fc9c17369e368dddc1fa61))
* remove useless entrypoint class ([957eab0](https://github.com/angelacorte/fieldVMC/commit/957eab0ccafae91b58ca767222dd53e8a2f359c8))
* stick to classic VMC resources quantity ([218d4af](https://github.com/angelacorte/fieldVMC/commit/218d4afc05a750fef409542e6a479c3dbc045f44))
* try to fix main logic ([0587276](https://github.com/angelacorte/fieldVMC/commit/0587276acc2d65bd1733a4569443e39c00e1352a))
* weight should be double ([f3bfbc2](https://github.com/angelacorte/fieldVMC/commit/f3bfbc2dcb4d34ded081305160c3c5faaa7e2637))

### General maintenance

* add effects for classic vmc ([9b8bd8d](https://github.com/angelacorte/fieldVMC/commit/9b8bd8d9f71774bd8fd77ba30be46532270bc8e6))
* add gitignore ([adb7b2a](https://github.com/angelacorte/fieldVMC/commit/adb7b2ac8831d16e4588871b851eba11886b5a95))
* add preliminary data on fixed leader ([61c9bd1](https://github.com/angelacorte/fieldVMC/commit/61c9bd1c3a030dac11e49a37b229f3d43cb13539))
* **ci:** add check simulations docker ([b1348e7](https://github.com/angelacorte/fieldVMC/commit/b1348e7a7a7b71c5b505f5f57896b39b09c8a791))
* **release:** update gradle.properties .env versions to 1.2.0 [skip ci] ([c8e153e](https://github.com/angelacorte/fieldVMC/commit/c8e153e73042753d846be1af5f15ec6aa179ef16))
* update gitingore ([f86d027](https://github.com/angelacorte/fieldVMC/commit/f86d027909bb2b149fcbcae48833cb238a3d173d))

### Style improvements

* remove unused imports ([dce8669](https://github.com/angelacorte/fieldVMC/commit/dce8669538be71034b8610b0574247a054518ce2))

### Refactoring

* move common functions into proper package ([88fb10a](https://github.com/angelacorte/fieldVMC/commit/88fb10acbd4103a7fb7eb5380f750cc0e0056ca0))
* rename clock/execution clock ([77b0c01](https://github.com/angelacorte/fieldVMC/commit/77b0c01191b86abbac775d6a5bb3bbe2c2a02141))

## [1.2.0](https://github.com/angelacorte/fieldVMC/compare/1.1.10...1.2.0) (2024-12-04)

### Features

* remove all context receivers ([098e154](https://github.com/angelacorte/fieldVMC/commit/098e154c2c2ebbb1330fe2bbf2aaa7d0ebcff36b))

### Dependency updates

* **deps:** update dependency matplotlib to v3.9.3 ([bee86b4](https://github.com/angelacorte/fieldVMC/commit/bee86b4e60f2073fa1bd97ba6b62b22524d9af73))

### Bug Fixes

* add missing parameter in node properties ([a1c005e](https://github.com/angelacorte/fieldVMC/commit/a1c005e425f11220fff2e9a515caf70b4c69da97))
* add missing parameter in node properties ([b5000b5](https://github.com/angelacorte/fieldVMC/commit/b5000b53db1eb3ebd35c91f55161eb3d5776d75d))
* correct use of distance sensor in context ([caebdbc](https://github.com/angelacorte/fieldVMC/commit/caebdbc3e11cca2437c2d6b59cd9ca76cbdd0e8c))
* error when using distances() ([a5a1bf0](https://github.com/angelacorte/fieldVMC/commit/a5a1bf0e397cdc1e853a7dd122ca4ec49434ba8b))
* generating again all simulations instead of just the fixed leader one ([94e30b7](https://github.com/angelacorte/fieldVMC/commit/94e30b762f72101ffb2c3b5a07cd679398f02530))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.10 [skip ci] ([cd44f59](https://github.com/angelacorte/fieldVMC/commit/cd44f59ce8cd0a22b4176cd81c2f184cd86b31bf))

### Refactoring

* move common functions into another file ([c6fd8ab](https://github.com/angelacorte/fieldVMC/commit/c6fd8abcc019a907c5a7300cd2af581a13725b6f))
* remove duplicate code ([63697f4](https://github.com/angelacorte/fieldVMC/commit/63697f416394c0f96bb46067e9ec3d9af2acdbef))
* use stdlib's distanceTo function ([6239c45](https://github.com/angelacorte/fieldVMC/commit/6239c4598994d225abadbd7adf0a5b43f87b8014))

## [1.1.10](https://github.com/angelacorte/fieldVMC/compare/1.1.9...1.1.10) (2024-12-02)

### Dependency updates

* **core-deps:** update collektive to v11.0.2 ([1dfdfa5](https://github.com/angelacorte/fieldVMC/commit/1dfdfa5cf5045eb743d3a274158cdf671bc8082c))
* **deps:** update plugin org.gradle.toolchains.foojay-resolver-convention to v0.9.0 ([afda1da](https://github.com/angelacorte/fieldVMC/commit/afda1da51c4d3c3d9dcd80a26b3141552fe14f59))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.9 [skip ci] ([b57c445](https://github.com/angelacorte/fieldVMC/commit/b57c44568739118cf591033deb4a7cd7036c858f))

## [1.1.9](https://github.com/angelacorte/fieldVMC/compare/1.1.8...1.1.9) (2024-11-30)

### Dependency updates

* **core-deps:** update collektive to v11.0.1 ([fc38232](https://github.com/angelacorte/fieldVMC/commit/fc38232996c0e8142e0ee6c2cbb500f5612608b5))
* **deps:** update alchemist to v36.0.2 ([301c9f2](https://github.com/angelacorte/fieldVMC/commit/301c9f29fcd3f0104e7057f2ce1ffae75818f6e0))
* **deps:** update alchemist to v36.0.3 ([fcf1a58](https://github.com/angelacorte/fieldVMC/commit/fcf1a5834589c6d4bccc60d79f62419843162dbe))
* **deps:** update dependency it.unibo.alchemist:alchemist-ui-tooling to v36.0.1 ([b8b96fd](https://github.com/angelacorte/fieldVMC/commit/b8b96fde5f527def84445df67347857e2330712d))
* **deps:** update plugin kotlin-qa to v0.75.0 ([6a06a62](https://github.com/angelacorte/fieldVMC/commit/6a06a620105e69dd29ce55c036f7d2f1b465db1d))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.8 [skip ci] ([0b54f7d](https://github.com/angelacorte/fieldVMC/commit/0b54f7d8b2af2ecadc6f2a69540e5f21cfcc5348))

## [1.1.8](https://github.com/angelacorte/fieldVMC/compare/1.1.7...1.1.8) (2024-11-27)

### Dependency updates

* **core-deps:** update collektive to v11 ([cfabec6](https://github.com/angelacorte/fieldVMC/commit/cfabec67a3a68d53271939320b172b8cb6ef115b))
* **deps:** update dependency gradle to v8.11 ([b8cef7c](https://github.com/angelacorte/fieldVMC/commit/b8cef7ce38bd6e931e76ca7b41a4d737647a0a05))
* **deps:** update dependency gradle to v8.11.1 ([31069e3](https://github.com/angelacorte/fieldVMC/commit/31069e3abffdfb402ba7b237c5ae6ed189d2ef14))
* **deps:** update dependency it.unibo.alchemist:alchemist to v35.0.1 ([694c419](https://github.com/angelacorte/fieldVMC/commit/694c419f18179b26a61818eeb190d325107e92b4))
* **deps:** update dependency it.unibo.alchemist:alchemist-euclidean-geometry to v35.0.2 ([cfc6d64](https://github.com/angelacorte/fieldVMC/commit/cfc6d643449a35270092541e3b7c43cbfccc35c2))
* **deps:** update dependency it.unibo.alchemist:alchemist-swingui to v36 ([b2f2dad](https://github.com/angelacorte/fieldVMC/commit/b2f2dadd59f5a4a8cd7807206155ddc9f223ff74))
* **deps:** update dependency it.unibo.alchemist:alchemist-ui-tooling to v35.0.3 ([6fefa6e](https://github.com/angelacorte/fieldVMC/commit/6fefa6ea973c7f4be4a7a2d8387123af1b137c65))
* **deps:** update dependency xarray to v2024.11.0 ([e7e9e3a](https://github.com/angelacorte/fieldVMC/commit/e7e9e3a8c65e514eef862a15fe08eaa55b6b5002))
* **deps:** update plugin kotlin-qa to v0.70.0 ([fe16453](https://github.com/angelacorte/fieldVMC/commit/fe164537e28f016cf3e0b80008136195f81f2217))
* **deps:** update plugin kotlin-qa to v0.70.1 ([b5f327a](https://github.com/angelacorte/fieldVMC/commit/b5f327a73453e57c4f3413423a28a983832645a3))
* **deps:** update plugin kotlin-qa to v0.70.2 ([a758ac4](https://github.com/angelacorte/fieldVMC/commit/a758ac465a80a96652f68c766e13d3c998a57467))
* **deps:** update plugin kotlin-qa to v0.71.0 ([e25d951](https://github.com/angelacorte/fieldVMC/commit/e25d9517d207444ff63c13efe6d20fac2247c2b7))
* **deps:** update plugin kotlin-qa to v0.73.0 ([03cf253](https://github.com/angelacorte/fieldVMC/commit/03cf253d616544014f81d1c2265358eaf9a098b5))
* **deps:** update plugin kotlin-qa to v0.74.0 ([ca47910](https://github.com/angelacorte/fieldVMC/commit/ca4791079fbfcc0b538c5e6f5d9bd9d489a7c8fe))
* **deps:** update plugin multijvmtesting to v1.3.1 ([5a867df](https://github.com/angelacorte/fieldVMC/commit/5a867dfd1501bf339f8e184001387183266b10a5))
* **deps:** update plugin multijvmtesting to v1.3.2 ([b096e11](https://github.com/angelacorte/fieldVMC/commit/b096e111e73427a41781d2db0f392635296c86e0))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.15 ([42c3e73](https://github.com/angelacorte/fieldVMC/commit/42c3e737338f8cb60b1c2f16f82cb48cd3893e69))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.16 ([77d5427](https://github.com/angelacorte/fieldVMC/commit/77d542712618d3e08d783a7b4f3490d70ac2d6b5))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.17 ([71109f9](https://github.com/angelacorte/fieldVMC/commit/71109f93965e7f5e2dcfca3e51d20369fc5e6924))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.18 ([419745e](https://github.com/angelacorte/fieldVMC/commit/419745e7f04c1d59fd5abc22ad5cacd88093496a))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.19 ([00e8a7f](https://github.com/angelacorte/fieldVMC/commit/00e8a7f5b6acb81c2bba5ee34bba3130d25b6aa7))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.20 ([34dd5e6](https://github.com/angelacorte/fieldVMC/commit/34dd5e6f873af451fff36e0d50d5eea37fc087fc))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.21 ([eb817b2](https://github.com/angelacorte/fieldVMC/commit/eb817b26b999782054d72fb50e9650c50a2ee7f7))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.22 ([e7053f9](https://github.com/angelacorte/fieldVMC/commit/e7053f9381f26fe3957fd7498164b229be87383a))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.23 ([b807010](https://github.com/angelacorte/fieldVMC/commit/b807010f729cce779e29977cd1d78a7b08600cca))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.7 [skip ci] ([94fc929](https://github.com/angelacorte/fieldVMC/commit/94fc929cb5206592c7002eaa05fda565c89c4c51))

### Refactoring

* update repeat into evolve as in Collektive v11 ([a75a156](https://github.com/angelacorte/fieldVMC/commit/a75a1564725b9e4c8c7a2404269eae0701a7210c))

## [1.1.7](https://github.com/angelacorte/fieldVMC/compare/1.1.6...1.1.7) (2024-11-14)

### Dependency updates

* **core-deps:** update dependency it.unibo.collektive.collektive-plugin to v10.11.1 ([cddaf12](https://github.com/angelacorte/fieldVMC/commit/cddaf1210e98fc67f5a95230167014d1abcc2238))
* **deps:** update alchemist to v35 ([79daca0](https://github.com/angelacorte/fieldVMC/commit/79daca0966f28a688da788f055fc8c71881027b3))
* **deps:** update plugin com.gradle.enterprise to v3.18.2 ([8d84014](https://github.com/angelacorte/fieldVMC/commit/8d840146479c6b047ed4b02121952dd3adcf05e7))

### Build and continuous integration

* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.13 ([3a43b54](https://github.com/angelacorte/fieldVMC/commit/3a43b549332c36104a238b5b62a05b4449b56821))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3.5.14 ([aa49c30](https://github.com/angelacorte/fieldVMC/commit/aa49c3023b76967ea0f6188426ff0b81c3e689d5))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.6 [skip ci] ([2b0b6e2](https://github.com/angelacorte/fieldVMC/commit/2b0b6e2adb2ca69791f942c4dc411768835ff942))

## [1.1.6](https://github.com/angelacorte/fieldVMC/compare/1.1.5...1.1.6) (2024-11-12)

### Dependency updates

* **core-deps:** update collektive to v10.11.0 ([32769b4](https://github.com/angelacorte/fieldVMC/commit/32769b4c7d85cfb37faa7faff9d8f04634dcb55d))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.5 [skip ci] ([52f34a1](https://github.com/angelacorte/fieldVMC/commit/52f34a1474ba06cef27cc88f24b9a45b1a632358))

## [1.1.5](https://github.com/angelacorte/fieldVMC/compare/1.1.4...1.1.5) (2024-11-11)

### Dependency updates

* **core-deps:** update collektive to v10.10.0 ([68d4a8f](https://github.com/angelacorte/fieldVMC/commit/68d4a8f504af7dfce776f34e8d167a9759459de7))
* **deps:** update alchemist to v34.1.14 ([68d8f9f](https://github.com/angelacorte/fieldVMC/commit/68d8f9fd51a787f482b5f49277899bf29829e3d3))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.115 ([8c38608](https://github.com/angelacorte/fieldVMC/commit/8c38608b2420ebbc2cd79990dcb914bb23c45192))
* **deps:** update plugin multijvmtesting to v1.3.0 ([ea36123](https://github.com/angelacorte/fieldVMC/commit/ea3612384d3463447731b70fca43dd295afa8d64))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.4 [skip ci] ([0919444](https://github.com/angelacorte/fieldVMC/commit/0919444cd20f083817019ff881aea869bb1707a3))

## [1.1.4](https://github.com/angelacorte/fieldVMC/compare/1.1.3...1.1.4) (2024-11-08)

### Dependency updates

* **core-deps:** update collektive to v10.9.0 ([fcffeb6](https://github.com/angelacorte/fieldVMC/commit/fcffeb617086d7dac6da52ce0f58b76d73aee3c5))
* **deps:** update plugin multijvmtesting to v1 ([98e63cb](https://github.com/angelacorte/fieldVMC/commit/98e63cbc85cc62987248e1627e12f6da128c95c8))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.3 [skip ci] ([c1e6f57](https://github.com/angelacorte/fieldVMC/commit/c1e6f574595e30a5a998b480f06b90037fb65c92))

## [1.1.3](https://github.com/angelacorte/fieldVMC/compare/1.1.2...1.1.3) (2024-11-07)

### Dependency updates

* **core-deps:** update collektive to v10.7.0 ([f18351d](https://github.com/angelacorte/fieldVMC/commit/f18351d6301e6f27f5aea9b9ab2901ff558f728d))
* **deps:** update dependency gradle to v8.10.2 ([8a1a064](https://github.com/angelacorte/fieldVMC/commit/8a1a06409adbb9685a6f1eebe47f2e31ced46183))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.114 ([#5](https://github.com/angelacorte/fieldVMC/issues/5)) ([64e4456](https://github.com/angelacorte/fieldVMC/commit/64e445640193bb6bebfad76febc4cc87a2db37b8))
* **deps:** update node.js to v22 ([#27](https://github.com/angelacorte/fieldVMC/issues/27)) ([fd9f720](https://github.com/angelacorte/fieldVMC/commit/fd9f720d3079660a6f6c214621909c4e762a4be4))
* **deps:** update plugin com.gradle.enterprise to v3.18.1 ([3628423](https://github.com/angelacorte/fieldVMC/commit/3628423076c19c10204d0356a79e80c5fe85489f))
* **deps:** update plugin kotlin-qa to v0.61.1 ([#7](https://github.com/angelacorte/fieldVMC/issues/7)) ([4c825f5](https://github.com/angelacorte/fieldVMC/commit/4c825f54348960ed53f89c1c4958d20765260073))
* **deps:** update plugin kotlin-qa to v0.69.0 ([7d139c5](https://github.com/angelacorte/fieldVMC/commit/7d139c5e8d353f1572e6821e2cbe7a7be80c0cf6))

### Build and continuous integration

* **deps:** update actions/download-artifact action to v4.1.8 ([12173ee](https://github.com/angelacorte/fieldVMC/commit/12173ee4376299dbbcf00579e847e522b1b1d313))
* **deps:** update actions/setup-python action to v5.3.0 ([0def8f0](https://github.com/angelacorte/fieldVMC/commit/0def8f088d3ffd57d2d63125455869a8135166cb))
* **deps:** update danysk/build-check-deploy-gradle-action action to v3 ([015c115](https://github.com/angelacorte/fieldVMC/commit/015c115d959fa1f66e671b217abce5e6e33823d4))

### General maintenance

* **release:** update gradle.properties .env versions to 1.1.2 [skip ci] ([a324167](https://github.com/angelacorte/fieldVMC/commit/a324167ed3715b2669dd2548227472541982ee10))

## [1.1.2](https://github.com/angelacorte/fieldVMC/compare/1.1.1...1.1.2) (2024-11-07)


### Dependency updates

* **core-deps:** update dependency org.jetbrains.kotlin.jvm to v2.0.21 ([d0bce08](https://github.com/angelacorte/fieldVMC/commit/d0bce08f0f85938f1bd3a167d5fb3a1a76cc0097))
* **deps:** update alchemist to v34.1.12 ([93ecc76](https://github.com/angelacorte/fieldVMC/commit/93ecc76309c8c3e5e77f37729e987a99237d7345))
* **deps:** update dependency matplotlib to v3.9.2 ([a8e990d](https://github.com/angelacorte/fieldVMC/commit/a8e990dd058379f12fd597e716bf6b0584841878))
* **deps:** update dependency xarray to v2024.10.0 ([565d130](https://github.com/angelacorte/fieldVMC/commit/565d130361770610084d7f49dfae6121b4343f71))
* **deps:** update node.js to 20.18 ([b42fbd3](https://github.com/angelacorte/fieldVMC/commit/b42fbd3e2f1d41c4a6efd347fa18d539befac1b3))
* **deps:** update plugin gitsemver to v3.1.7 ([292db26](https://github.com/angelacorte/fieldVMC/commit/292db261f2dd4471ade76547bec1e71ef95cc232))
* **deps:** update plugin tasktree to v4 ([#30](https://github.com/angelacorte/fieldVMC/issues/30)) ([b04d25e](https://github.com/angelacorte/fieldVMC/commit/b04d25ea8038559a285e5503dac0893ec7266b84))
* **deps:** update python docker tag to v3.13.0 ([#22](https://github.com/angelacorte/fieldVMC/issues/22)) ([05249b1](https://github.com/angelacorte/fieldVMC/commit/05249b192a66b68f9302bf178399fa0f100942ed))


### Build and continuous integration

* **deps:** update actions/checkout action to v4.2.2 ([2fce0a0](https://github.com/angelacorte/fieldVMC/commit/2fce0a0368a1b676354b331b8bfe55b8e9ad03bd))
* **deps:** update actions/setup-node action to v4.0.4 ([f471a00](https://github.com/angelacorte/fieldVMC/commit/f471a00fee8f1b4b6ce8183de8abd27d60760fa1))
* **deps:** update actions/setup-node action to v4.1.0 ([3fb7ff4](https://github.com/angelacorte/fieldVMC/commit/3fb7ff4ee0724715d31d9dfad29a5f7cf69da61b))
* **deps:** update actions/setup-python action to v5.1.1 ([#12](https://github.com/angelacorte/fieldVMC/issues/12)) ([3b7a10d](https://github.com/angelacorte/fieldVMC/commit/3b7a10da596515e535fb85de16cb1f36a9a74201))
* **deps:** update actions/upload-artifact action to v4.4.3 ([1ecb6b2](https://github.com/angelacorte/fieldVMC/commit/1ecb6b2c7d1db6d49ae8e38eabdb25f5ccb341a6))
* **deps:** update dependency ubuntu to v24 ([0223ca5](https://github.com/angelacorte/fieldVMC/commit/0223ca56adfaf8b29fb0984c5a4d2897380f84e7))


### General maintenance

* **release:** update gradle.properties .env versions to 1.1.1 [skip ci] ([16e08d3](https://github.com/angelacorte/fieldVMC/commit/16e08d31b674a87526aa03fed510626dc84987d8))

## [1.1.1](https://github.com/angelacorte/fieldVMC/compare/1.1.0...1.1.1) (2024-11-07)


### Dependency updates

* **core-deps:** update collektive to v10.5.3 ([#1](https://github.com/angelacorte/fieldVMC/issues/1)) ([d4f104d](https://github.com/angelacorte/fieldVMC/commit/d4f104d667e4f02b49d0264cdec5f7aa3c478e03))


### General maintenance

* **release:** update gradle.properties .env versions to 1.1.0 [skip ci] ([673d88d](https://github.com/angelacorte/fieldVMC/commit/673d88dc68e71a27f0017bf4729a0f252c4513ab))

## [1.1.0](https://github.com/angelacorte/fieldVMC/compare/1.0.0...1.1.0) (2024-11-07)


### Features

* add field vmc that spawns node from a fixed root with fixed resources ([fc071ca](https://github.com/angelacorte/fieldVMC/commit/fc071cad51319a4f7eb4c9a8c4ad024110edf9e8))
* add maximum resource and fixed leader to skip election in new vmc function ([9d49494](https://github.com/angelacorte/fieldVMC/commit/9d49494fe6327b2bccadd15a60262819d5ee1892))
* add new simulation yaml for fixed leader simulation ([7949e69](https://github.com/angelacorte/fieldVMC/commit/7949e693025911cf5b80f6287a9c6a0b11e4dde4))


### Bug Fixes

* mergify bot account ([9c1aec6](https://github.com/angelacorte/fieldVMC/commit/9c1aec644a0177591a8e685a18ecde387bd2b300))
* typo in mergify bot account ([d2b52ca](https://github.com/angelacorte/fieldVMC/commit/d2b52ca6ef55c467a76563dcc50ab46677ac067f))


### General maintenance

* **build:** build only the experiment with fixed leader ([bbbe240](https://github.com/angelacorte/fieldVMC/commit/bbbe240eac6d8169d5674293b0fc565b87240e12))
* **build:** remove warning as error ([6618f87](https://github.com/angelacorte/fieldVMC/commit/6618f871979aa10543b5a28129ce66874ef8fd61))
* **libs:** update libs versions ([787851b](https://github.com/angelacorte/fieldVMC/commit/787851b1674a7780f501bfd6bc33c561d332f929))
* **release:** update gradle.properties .env versions to 1.0.0 [skip ci] ([cfcdb21](https://github.com/angelacorte/fieldVMC/commit/cfcdb212d2e2ddc0e99d612377ebfd0a9fec1de8))
* temporary comment old YAMLs ([da02931](https://github.com/angelacorte/fieldVMC/commit/da02931e968fc34ba8567603beb4c68827a53eca))
* update mergify config ([5c0552d](https://github.com/angelacorte/fieldVMC/commit/5c0552d5fee322580b09147d17c296fb51b102ff))
* use Collektive's renovate configuration ([0204241](https://github.com/angelacorte/fieldVMC/commit/0204241fefebc592195c5521c8d62a191bfbaae8))


### Refactoring

* move utils functions in proper file ([5d3eec2](https://github.com/angelacorte/fieldVMC/commit/5d3eec214a67cd8485172f51bad5db3e37d650a4))
* use stdlib functions ([eb15dbd](https://github.com/angelacorte/fieldVMC/commit/eb15dbdec574a26aaea4344ca9581b5e3cd0633a))

## 1.0.0 (2024-10-23)


### Features

* first fieldVMC implementation proposed at ACSOS 2024 ([57f7c85](https://github.com/angelacorte/fieldVMC/commit/57f7c85e30c3a8dfbe06e0b23f5d9bc9e4b2ca28))


### Build and continuous integration

* temporary remove dockerhub login ([1fa0c35](https://github.com/angelacorte/fieldVMC/commit/1fa0c35812307f9acbe412086e68fd9eb9a0680f))


### General maintenance

* update gitignore ([741ec24](https://github.com/angelacorte/fieldVMC/commit/741ec247befcf8eaea79f9c9ddd91da034ab7b90))
* update project name ([427b68e](https://github.com/angelacorte/fieldVMC/commit/427b68e44fa9facb6948f30ff146aae4e3ddb856))

## [1.3.2](https://github.com/angelacorte/vmc-experiments/compare/1.3.1...1.3.2) (2024-08-02)


### Bug Fixes

* silence alchemist verbosity ([be520e2](https://github.com/angelacorte/vmc-experiments/commit/be520e20c53c8cc30f1b9a832cc4105c3abf69e1))
* update readme ([4f81d59](https://github.com/angelacorte/vmc-experiments/commit/4f81d59d7ecdec6db70d1b9dec07864908cf63f3))


### Documentation

* add documentation to main functions ([d139d51](https://github.com/angelacorte/vmc-experiments/commit/d139d51eba00e72cbc59cddafa2213b0fc839c5a))


### General maintenance

* **release:** update gradle.properties .env versions to 1.3.1 [skip ci] ([ffb2fc6](https://github.com/angelacorte/vmc-experiments/commit/ffb2fc69b99c55931069fa268c7fb3749f0fde7a))

## [1.3.1](https://github.com/angelacorte/vmc-experiments/compare/1.3.0...1.3.1) (2024-08-01)


### Documentation

* add documentation to main functions ([8cf4eef](https://github.com/angelacorte/vmc-experiments/commit/8cf4eef078abe9c7c0d2c298fb1f97e710033489))


### General maintenance

* **release:** update gradle.properties .env versions to 1.3.0 [skip ci] ([fc6baaf](https://github.com/angelacorte/vmc-experiments/commit/fc6baaf3ea811fb2c23a08e17a8e8dc44559b7d1))

## [1.3.0](https://github.com/angelacorte/vmc-experiments/compare/1.2.0...1.3.0) (2024-07-02)


### Features

* add variable to resource sensor ([911bc59](https://github.com/angelacorte/vmc-experiments/commit/911bc5917509d5da02ec7b6f4e14c65f8d13f63b))
* add variables to device spawner ([a4d7cb1](https://github.com/angelacorte/vmc-experiments/commit/a4d7cb1ee1442371275af61935f5ead4844ab801))


### Bug Fixes

* update readme ([9ee0c2d](https://github.com/angelacorte/vmc-experiments/commit/9ee0c2d0ec649696dec20b4fe12ec013d6e0b76e))


### General maintenance

* draft update on readme ([484605c](https://github.com/angelacorte/vmc-experiments/commit/484605ca5353fea4630f67010bd6355cacc11a36))
* fix formula ([0c09f2a](https://github.com/angelacorte/vmc-experiments/commit/0c09f2a053c15fd10fc911d28a6f300dace0f796))
* **release:** update gradle.properties .env versions to 1.2.0 [skip ci] ([f18b890](https://github.com/angelacorte/vmc-experiments/commit/f18b8904b28877680149819635db46d024f214ea))


### Refactoring

* remove comments ([ddf4d4b](https://github.com/angelacorte/vmc-experiments/commit/ddf4d4bfa03b84acf80e6992cbc4e113cf8ad340))
* remove useless parameters ([83f8b69](https://github.com/angelacorte/vmc-experiments/commit/83f8b6900c7031a24fd270298789acd76ddb5a5c))
* remove useless variables ([fce1dfb](https://github.com/angelacorte/vmc-experiments/commit/fce1dfbc18dc245ab64a227680a8ad4e1c78fc77))
* variables and entrypoints ([f3c671b](https://github.com/angelacorte/vmc-experiments/commit/f3c671b025f8d8b0979d39c8f679bea0a07ab94f))

## [1.2.0](https://github.com/angelacorte/vmc-experiments/compare/1.1.1...1.2.0) (2024-07-01)


### Features

* add quick-start script ([714198e](https://github.com/angelacorte/vmc-experiments/commit/714198e110864e799549a7912f508b84d22fa608))


### Bug Fixes

* correct toc ([097803c](https://github.com/angelacorte/vmc-experiments/commit/097803c7763c5a5de52c7a088121c7badaaa6cf3))
* remove docker commands ([a28fb0c](https://github.com/angelacorte/vmc-experiments/commit/a28fb0c2088470994c577b161e9bb392d70a749a))


### General maintenance

* draft update on readme ([c2d73c0](https://github.com/angelacorte/vmc-experiments/commit/c2d73c02f105d9771e18e66c97f793c2da0e9ec4))
* **release:** update gradle.properties .env versions to 1.1.1 [skip ci] ([5f79ab6](https://github.com/angelacorte/vmc-experiments/commit/5f79ab62dbf5dbb7ffb1304ad1d692910ea8a973))
* update README ([3586efa](https://github.com/angelacorte/vmc-experiments/commit/3586efa12571c216043c58a8e6df6c5890d6dd18))

## [1.1.1](https://github.com/angelacorte/vmc-experiments/compare/1.1.0...1.1.1) (2024-05-15)


### Bug Fixes

* simulator zoom ratio ([72e5d1f](https://github.com/angelacorte/vmc-experiments/commit/72e5d1fa42244c5613db5040d145c1f7bddd5cb0))


### General maintenance

* **release:** update gradle.properties .env versions to 1.1.0 [skip ci] ([7ec812f](https://github.com/angelacorte/vmc-experiments/commit/7ec812f9cd1aa05bb4f835317d4a99e75e77e1d0))
* update README ([06180b9](https://github.com/angelacorte/vmc-experiments/commit/06180b9017ef05cd3257181f0985a984949c0607))

## [1.1.0](https://github.com/angelacorte/vmc-experiments/compare/1.0.0...1.1.0) (2024-05-11)


### Features

* add simulation with multi-leader election ([#1](https://github.com/angelacorte/vmc-experiments/issues/1)) ([aa4b7e8](https://github.com/angelacorte/vmc-experiments/commit/aa4b7e8d715fe89268cac562c7aa3af1c30f0b38))
* first draft for readme ([efc77b1](https://github.com/angelacorte/vmc-experiments/commit/efc77b1f156b0eb52773c7e2b87852757dadb6a9))


### Bug Fixes

* ci release ([2b3441c](https://github.com/angelacorte/vmc-experiments/commit/2b3441c06d682f067562d001f3fccfe09fe188e4))
* ci release ([e4be52c](https://github.com/angelacorte/vmc-experiments/commit/e4be52c3a66de215918d51046c56dd77621fcf05))


### General maintenance

* add images for readme ([490b96f](https://github.com/angelacorte/vmc-experiments/commit/490b96f375c572da6f36fd78874dbfb0b2357d2f))
* **release:** update gradle.properties .env versions to 1.0.0 [skip ci] ([888ecc5](https://github.com/angelacorte/vmc-experiments/commit/888ecc503343fda3268702165b37713a565823bb))

## 1.0.0 (2024-05-11)


### Features

* add action for cloning nodes ([e3a615d](https://github.com/angelacorte/vmc-experiments/commit/e3a615d249fdd42aace8bd6d9405d2b1b8d226a9))
* add check on childs spawning limit ([4312245](https://github.com/angelacorte/vmc-experiments/commit/43122450675eea1911607982ef7a787092afd5da))
* add converge cast ([c7ed5a2](https://github.com/angelacorte/vmc-experiments/commit/c7ed5a2553a0cb4c7c3c85f31e87214ebc16e2da))
* add converge cast and spread to children ([1f6b4bd](https://github.com/angelacorte/vmc-experiments/commit/1f6b4bd3de90c6460dbc000dbeffdadd600ea15d))
* add costants for formulas ([fab3ea1](https://github.com/angelacorte/vmc-experiments/commit/fab3ea1c3e9837a823cb4df065d23b07a2d5f070))
* add distancebetween function ([7528a7a](https://github.com/angelacorte/vmc-experiments/commit/7528a7a8e090dcfd978f3526d3460e7bb5058b7e))
* add function to set the leader id ([62f8952](https://github.com/angelacorte/vmc-experiments/commit/62f89523614698af4dbe9f143663de559a7f886b))
* add more parameters ([cbc2c7c](https://github.com/angelacorte/vmc-experiments/commit/cbc2c7cafee8437a9e45dcaec6985c26b1132c06))
* add random generator ([be37bf3](https://github.com/angelacorte/vmc-experiments/commit/be37bf366de25f2b07125e6fd401557b59fc0d31))
* add random sensor ([c2fb6a3](https://github.com/angelacorte/vmc-experiments/commit/c2fb6a35ccde08ba9d02ba5bfce63ae2258cbb65))
* add resource upper bound ([8279b93](https://github.com/angelacorte/vmc-experiments/commit/8279b939adb6d03df086d05ba5467a44d30d9d97))
* add seed for random and layer ([5c9f058](https://github.com/angelacorte/vmc-experiments/commit/5c9f058e0b7122303fc16e63d4b36e744d053da3))
* add sensor interface ([16754cb](https://github.com/angelacorte/vmc-experiments/commit/16754cba2b9f47af9939dae96ec134556f0d7273))
* add simulation for converge and spread ([e4b200e](https://github.com/angelacorte/vmc-experiments/commit/e4b200e36b91734f9650147fb8548d611f7dcd01))
* add vmc's draft algorithm ([dcfa791](https://github.com/angelacorte/vmc-experiments/commit/dcfa791a1fdd148bef449b89f8f1d44c73e8b972))
* change method for setting molecules ([3366426](https://github.com/angelacorte/vmc-experiments/commit/336642676c169554eccb48163a8872f0ddd3365a))
* change simulation entrypoint ([cf699dc](https://github.com/angelacorte/vmc-experiments/commit/cf699dcb45af281c528d8ff01e398de1b0427557))
* change simulation's parameters ([255a97a](https://github.com/angelacorte/vmc-experiments/commit/255a97aeabcd927b9361b24ce74553749bdfc5d0))
* create clone action ([4db27e5](https://github.com/angelacorte/vmc-experiments/commit/4db27e5127dcf112f0fcfb6cd13ac2b70d9a393a))
* create device destroyer sensor ([80c57b6](https://github.com/angelacorte/vmc-experiments/commit/80c57b66e9499559a861be4a11472215997749ff))
* create device spawner sensor ([25acfc9](https://github.com/angelacorte/vmc-experiments/commit/25acfc9d425c10d2989a5d42379eb085a522f83d))
* create device's location sensor ([42c3296](https://github.com/angelacorte/vmc-experiments/commit/42c3296b57cfe67dc5affa9d36646b4e7ede89d7))
* create entrypoint for converge and spread ([2063c7f](https://github.com/angelacorte/vmc-experiments/commit/2063c7fc0ecc47662e9db0e7a8c77eb783abc28d))
* create entrypoint for vmc simulation ([7f25c5c](https://github.com/angelacorte/vmc-experiments/commit/7f25c5cece24c594946c594adc476d00619b5917))
* create gradient cast and distanceTo ([67a42b8](https://github.com/angelacorte/vmc-experiments/commit/67a42b8e94aa92e27cddcec0f537a4487e2ff4f4))
* create leader sensor ([f0d5999](https://github.com/angelacorte/vmc-experiments/commit/f0d5999153e14ea1a4267cc695c71ea43e1c4187))
* create light sensor ([8b30729](https://github.com/angelacorte/vmc-experiments/commit/8b307297c06b7e4b5e8d5d3bb79a5255b7932072))
* create proper modules for program concepts ([57a1413](https://github.com/angelacorte/vmc-experiments/commit/57a14136e6c25d0eb1988a2b10654901028ecc6d))
* create resource sensor ([c7a1655](https://github.com/angelacorte/vmc-experiments/commit/c7a1655229da1083176847ec85bc21a98be161b8))
* create success sensor ([b00bdb0](https://github.com/angelacorte/vmc-experiments/commit/b00bdb0bc88c4bf140772bdea027973516455ca7))
* create success source layer ([acb1352](https://github.com/angelacorte/vmc-experiments/commit/acb13527e2ab5e5691f1bb7a9b94bcfa727d15c5))
* delete useless cloning action ([e3747ea](https://github.com/angelacorte/vmc-experiments/commit/e3747ea691663225dd9964f6f190d766d21042bd))
* get variables from environment ([7136152](https://github.com/angelacorte/vmc-experiments/commit/7136152a9e5a577a56a0d9e2ae85784a90ae4c8a))
* implement spawn based on childs number ([2e5edc8](https://github.com/angelacorte/vmc-experiments/commit/2e5edc849935a627cf0fc0b139888ce55aa98742))
* introduce layer property management ([407b947](https://github.com/angelacorte/vmc-experiments/commit/407b9475ef155d8b1c617fcfe6362ffa645af86d))
* parametrized initial nodes value ([7e8ee09](https://github.com/angelacorte/vmc-experiments/commit/7e8ee09a4d049d31a75f91071e743d3433608f52))
* remove simulation template ([39b11ea](https://github.com/angelacorte/vmc-experiments/commit/39b11ea70c763727b06620d9b41344a5da2b6a80))
* remove useless classes ([e993d64](https://github.com/angelacorte/vmc-experiments/commit/e993d6447eb1f29ea0b051eda24080f548288511))
* remove useless node contents clone ([81c1a18](https://github.com/angelacorte/vmc-experiments/commit/81c1a18b2c8a27c3c669c5a36b727d09cbdf7dfa))
* remove useless sensors ([2363ec2](https://github.com/angelacorte/vmc-experiments/commit/2363ec2a63f63aa46ceec4045aa17d5bc8087abf))
* rename block c into converge cast ([b0ab0c5](https://github.com/angelacorte/vmc-experiments/commit/b0ab0c566b87fef740f680445c8c20dcdb3d1ae0))
* rename random class ([2e51531](https://github.com/angelacorte/vmc-experiments/commit/2e515315c0cda875482c04cb2f773d68d349c826))
* run program with generic contexts ([9699070](https://github.com/angelacorte/vmc-experiments/commit/9699070a8ad33b3e2c522d7d25c24477d8c28852))
* running simulations with spawning, destroying and not ([47b5914](https://github.com/angelacorte/vmc-experiments/commit/47b5914276b0cc26cfb52c6a1d72a864b05587ec))
* spawn nodes farther from parent ([241726a](https://github.com/angelacorte/vmc-experiments/commit/241726a1aacba27c4509a6b43464ea1c3118a992))
* update yaml configuration ([1ca5896](https://github.com/angelacorte/vmc-experiments/commit/1ca58969f90bb1052537a9221e3b3601c210ebf9))
* use new contexts instead of environment variables ([261114b](https://github.com/angelacorte/vmc-experiments/commit/261114ba92e9cc7be493110cec440264dcc8ac5c))
* use new properties instead of molecules ([bc5d8c6](https://github.com/angelacorte/vmc-experiments/commit/bc5d8c6a09d13b434a56acf327c426f674dd23de))
* use new sensors ([9d4e783](https://github.com/angelacorte/vmc-experiments/commit/9d4e7830cb505b39eecd7667ceab4dc5db2bf70f))
* using Bidimensional gaussian layer ([38520dc](https://github.com/angelacorte/vmc-experiments/commit/38520dc556d736b1f09548bb2d39d3bb229d56fd))
* using bounded election ([866202e](https://github.com/angelacorte/vmc-experiments/commit/866202e9d883ff26f76a4d30704abf30e70f6845))
* using environment variables for setting molecules ([e9c7cd1](https://github.com/angelacorte/vmc-experiments/commit/e9c7cd1fbe1005bc4aa2192350c6282fcee0a50d))
* working bounded election ([995c008](https://github.com/angelacorte/vmc-experiments/commit/995c008b43e59c4fc8a9ca2c3651b1d3123b963c))


### Dependency updates

* **deps:** add collektive and alchemist dependencies ([cca7e3f](https://github.com/angelacorte/vmc-experiments/commit/cca7e3f396c9b1c913692bfc67dbe69f49c7d4ea))
* **deps:** add compiler plugin dependency and context receivers ([4c239f1](https://github.com/angelacorte/vmc-experiments/commit/4c239f198f466e149d9890dbb7d97d507675eb8e))
* **deps:** add kotest depencencies ([789ed52](https://github.com/angelacorte/vmc-experiments/commit/789ed52a6c7c6b66d01dc0803dd27a3feda21455))
* **deps:** update collektive version ([1d36b87](https://github.com/angelacorte/vmc-experiments/commit/1d36b870799006c84db2bfe8555c869bd20efc72))
* **deps:** update collektive version ([1e00dfb](https://github.com/angelacorte/vmc-experiments/commit/1e00dfba871edbd01d1bbc49e0525ca84cb26f68))


### Bug Fixes

* add correct parameter to property ([f117c32](https://github.com/angelacorte/vmc-experiments/commit/f117c32f566149101b432b717a3b9110a6481b8f))
* change arbitrary variable value ([20d3782](https://github.com/angelacorte/vmc-experiments/commit/20d37825512d12078e9f7de2d361cfcf53a08d8a))
* change docker username ([0138ae8](https://github.com/angelacorte/vmc-experiments/commit/0138ae83b139c55a003ad0bb380134c1571ff47a))
* correct molecule name ([8b839ad](https://github.com/angelacorte/vmc-experiments/commit/8b839adba2fdbe0e62177b5a35802fedeca4cf31))
* get initial resource as parameter ([0bfe2de](https://github.com/angelacorte/vmc-experiments/commit/0bfe2de43e90ae280b1f852672e19e3741742451))
* ignore branches ([019a274](https://github.com/angelacorte/vmc-experiments/commit/019a27456ac1f222a1cd45b1d20e2b8509f3ce8b))
* remove useless aggregate context ([dc044f6](https://github.com/angelacorte/vmc-experiments/commit/dc044f694c5ceaed7f733b112599ee5c42fe2081))
* typo in parameter ([4f546cf](https://github.com/angelacorte/vmc-experiments/commit/4f546cf8690700173dd7be4cf816671051e3b265))


### General maintenance

* add effects for converge and spread simulation ([1f1145f](https://github.com/angelacorte/vmc-experiments/commit/1f1145f0d780db0d303e5e37c68d84788d0a5b46))
* add kotlin reflect ([915a0bb](https://github.com/angelacorte/vmc-experiments/commit/915a0bbcedaff3dd65f52682603af11c1bf79b0d))
* add setting molecule ([665788e](https://github.com/angelacorte/vmc-experiments/commit/665788e3d5b3137d2722ee5f859bd1734553c9fe))
* change project name ([f1631ac](https://github.com/angelacorte/vmc-experiments/commit/f1631ac21ae11d8c7a2438e2dd316b5bb8a1c3b6))
* remove check simulations docker ([cf27cce](https://github.com/angelacorte/vmc-experiments/commit/cf27cceef64950f6f100ed9db683f8106091dc61))
* remove print ([f228fad](https://github.com/angelacorte/vmc-experiments/commit/f228fad5273903d03bfc52b24ec1a6c36fc2d696))
* remove unused imports ([061c3d5](https://github.com/angelacorte/vmc-experiments/commit/061c3d5e4aa2b0fa140d1aebc35970876cd5e21d))
* update effects ([0b81829](https://github.com/angelacorte/vmc-experiments/commit/0b81829b94a16ed4a64d2eb333ae394e0b3acb22))


### Refactoring

* delete unused simulations ([9eb7338](https://github.com/angelacorte/vmc-experiments/commit/9eb7338f231bb94650487ff749b089c4fc324b13))
* modulize program ([ba76a88](https://github.com/angelacorte/vmc-experiments/commit/ba76a883683d7b26b0a07e01c0f095abd1197fde))
* move classes into proper package ([15f192d](https://github.com/angelacorte/vmc-experiments/commit/15f192dd8ecef8d470d4c6db7f8f9e0efc7ec50f))
* moved my hop metric ([cff8cb2](https://github.com/angelacorte/vmc-experiments/commit/cff8cb2bd6c404e8172ccc1fe805ce1ed2d24115))

## [1.2.1](https://github.com/DanySK/alchemist-experiments-bootstrap/compare/1.2.0...1.2.1) (2024-02-09)


### Dependency updates

* **deps:** update alchemist to v25.16.2 ([153a932](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/153a932ad1c712b2ba41304fb1c68ddff8331564))
* **deps:** update alchemist to v26 ([079e0c9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/079e0c9f44e32d2309e22946e6a89cb0afb7338b))
* **deps:** update alchemist to v26.0.1 ([aee7286](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/aee7286c958540a5c7ec846526cc2d9ed5732c04))
* **deps:** update alchemist to v26.0.10 ([8f5485e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8f5485ef904c283c9564d72ae3d2b7b3289d2dcf))
* **deps:** update alchemist to v26.0.11 ([8f584c4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8f584c45d9c577d8a9064e705c4ba29feaca7d3a))
* **deps:** update alchemist to v26.0.12 ([4a19ba6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4a19ba6b72fa51ac1a2b669b9c2a09edc4be47e1))
* **deps:** update alchemist to v26.0.13 ([22a0c75](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/22a0c7550f774babb6f27402564043f36c734357))
* **deps:** update alchemist to v26.0.2 ([0aa1060](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/0aa106079ce93c7060e45a379e4b0c2062c598da))
* **deps:** update alchemist to v26.0.3 ([7b359d6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7b359d6cdf9afef92a979d181d8d0db4620a9278))
* **deps:** update alchemist to v26.0.4 ([28a633f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/28a633f5f2235458873f7fb32291699123b768c5))
* **deps:** update alchemist to v26.0.5 ([9a738a1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9a738a1e190a5bd08232c23c194d11fff1148070))
* **deps:** update alchemist to v26.0.6 ([80431b2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/80431b21faa696d3dcf6a03d6e8fd26feecff3b6))
* **deps:** update alchemist to v26.0.7 ([1a749f1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1a749f1ac5a163d9a4188fc8d7a8605418b31158))
* **deps:** update alchemist to v29 ([c537825](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c5378254ffdfa58aa30809773e9750414b21ce08))
* **deps:** update alchemist to v29.3.5 ([811e29c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/811e29ca569aa18c09f40505c0239aeaccdfe9d7))
* **deps:** update alchemist to v29.4.0 ([b94c580](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b94c580399e9e4b0aee879d2abcd8bbc1cd0ecd6))
* **deps:** update alchemist to v29.5.0 ([7290c8e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7290c8eab5120d1c0d9142a4516ee6fa4d519339))
* **deps:** update alchemist to v29.5.1 ([bc7f3d8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bc7f3d8a98e1c526b9e6b3de23a25d857747377b))
* **deps:** update alchemist to v30 (major) ([#693](https://github.com/DanySK/alchemist-experiments-bootstrap/issues/693)) ([97a25af](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/97a25af6a75e90e1c1d40a71be533ef39351feec))
* **deps:** update alchemist to v30.0.3 ([ca003b0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ca003b0e491e40a895232cf211418118de57bc53))
* **deps:** update alchemist to v30.0.4 ([94ead24](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/94ead247aa558f5c1af58232c5212cd94a23448b))
* **deps:** update alchemist to v30.0.5 ([d7a6fd0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d7a6fd0655c3f1d4bb08e2122d059e319eaf8d14))
* **deps:** update alchemist to v30.1.2 ([ff9c311](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ff9c311accadeb89355dc4df50da2451fdd2d7b9))
* **deps:** update alchemist to v30.1.3 ([b4b8848](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b4b88482773b3075eb5492b57eebb3dab967abcc))
* **deps:** update alchemist to v30.1.4 ([d6339f9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d6339f9888e11d969aeda61f6ec0b8b50f4f9778))
* **deps:** update alpine docker tag to v3.18.0 ([ac0e871](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ac0e8711b8667e99e9ba13c9deb7d4da2b61f316))
* **deps:** update alpine docker tag to v3.18.2 ([58f0bc2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/58f0bc2e83ab7154b174b5cb12df7d582cf3d4bc))
* **deps:** update alpine docker tag to v3.18.3 ([dc2b5ab](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/dc2b5abc1a2c7f291400e49c4d73c1b6f0d25439))
* **deps:** update alpine docker tag to v3.18.4 ([5059cc1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5059cc120d5e561fc1433d5984940c2279363b16))
* **deps:** update alpine docker tag to v3.18.5 ([8e805c8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8e805c8ffd729a228db8ce992d91334665373b2e))
* **deps:** update alpine docker tag to v3.19.0 ([04d3edf](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/04d3edfe2341695f16a878118540626b92eeee7a))
* **deps:** update alpine docker tag to v3.19.1 ([bef6f8f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bef6f8f1697e54ec865ab244d6fd9fab837ee463))
* **deps:** update dependency gradle to v8.2 ([487d145](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/487d1454983fef42da79e6cac067901c393e8b23))
* **deps:** update dependency gradle to v8.2.1 ([a7efd0c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a7efd0cfc81668e6714b9018c4d65fe7e131b44e))
* **deps:** update dependency gradle to v8.3 ([e8164ad](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e8164ad4b78ae646ad3e55bedd96a5f0d1665df1))
* **deps:** update dependency gradle to v8.4 ([f043d9e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f043d9ea77e545b9411c1a4e8ab0716d8dc3a194))
* **deps:** update dependency gradle to v8.5 ([4f83a98](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4f83a986ce5c6f6d0a76566466667d0b553b025e))
* **deps:** update dependency gradle to v8.6 ([4ff0a76](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4ff0a767ef99170569f2bab640aef07d36978247))
* **deps:** update dependency it.unibo.alchemist:alchemist-incarnation-scafi to v26.0.8 ([f4defbd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f4defbd197f9efdc133a8f0e54c113b5d661ae95))
* **deps:** update dependency matplotlib to v3.7.2 ([acd80b6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/acd80b6a187dc595066c55407c7ad88117acd74b))
* **deps:** update dependency matplotlib to v3.7.3 ([a83cfd9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a83cfd9ded76c09035517f078ebd1dfb2805b690))
* **deps:** update dependency matplotlib to v3.8.0 ([01f7dbb](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/01f7dbbaf54c2848fe970a1091749a73ff001577))
* **deps:** update dependency matplotlib to v3.8.1 ([2794a24](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2794a24208a896ab6f6710fd4c4f3bf580737496))
* **deps:** update dependency matplotlib to v3.8.2 ([8eb1902](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8eb19020513a6c43d979dd33083ad906123e6f24))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.8.22 ([aa2030c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/aa2030c5f9b3811c8750536a39d39b930ca6f383))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.9.0 ([65f3afe](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/65f3afe08ba42942525676656d6399879de554f9))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.9.10 ([2ed4c77](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2ed4c77ef458ac787f6876c5fc1a5290b1b47149))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.9.20 ([6f2bf6a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6f2bf6ac0d760cace65f25613a70a04f673668b4))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.9.21 ([de2a140](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/de2a1402b6516f8f03b17e910038d163f14568bb))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.9.22 ([9683dd5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9683dd595694a5f6a4c069afac7251a8802d15dd))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.27 ([fd1e295](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fd1e295c685ae9393a788c3647e5af68d965c13d))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.28 ([96438a4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/96438a441a6bc9a430380214e145c614c1db963e))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.29 ([a90ff87](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a90ff8797e5b6e7693fdb90ff57c8241df2212b2))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.30 ([d68da2c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d68da2c1d531a1e13d789399f993f4c62a8692ee))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.32 ([ae5d0ad](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ae5d0addb6fdf72c0b292087b005353f5a6bc0de))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.33 ([f6a98f4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f6a98f4f3ba236d2d8743362fa46ede5cc643d95))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.34 ([7a3a3af](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7a3a3af96f6114d54a3c3caeb60970b52d38dd4a))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.35 ([670b574](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/670b574e4c6d8f7b2c5c02da4d3efa8d27c8a0c1))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.36 ([d394a80](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d394a8074d7ba2bf2cd66d1c2a44dd81891a6387))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.37 ([0367dad](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/0367dad9fc0e1ff47dbb7e2cc576cfc28fb8cca9))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.38 ([abf257e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/abf257eccd6ff61c86f27f56a6d5f63292f711fe))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.39 ([885ce5a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/885ce5a691f5788cbcde60fc03d673a03294cf2f))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.40 ([4f9a341](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4f9a3415f1f7b2bdbc8e7d59b350c0fec49d640a))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.41 ([e2619ce](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e2619cea15b60b486f19c1d82709f2d882635959))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.42 ([a6af674](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a6af6745cc5957187d59c02d69bccb1a85c33ddc))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.43 ([c480c79](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c480c79bfae43907c826c6e08d707e22d638ef7a))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.44 ([eee738d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/eee738d16a94d952ccd3221f81aa770df977c15e))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.45 ([7ffb4d1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7ffb4d15884a2a18da876e684d63effa5c0306b1))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.46 ([89e9538](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/89e9538a0c01f673b1b318782889aac033dc72b2))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.47 ([3af7b30](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/3af7b3014e3b49ed159cb5e655898facdb0318bf))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.48 ([0d64917](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/0d649172834fc14cf101d2bb90f70471ff5fe007))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.49 ([ea84c04](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ea84c042761c84a846095df0b6ec10c2ad8ba6fe))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.50 ([3264507](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/32645078cc7539900ee7a78d2ffad3b1c27a2f1a))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.51 ([8c15c05](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8c15c057e10b775f63f22d9d4c2531b3d5083404))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.52 ([943449e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/943449e455962d4beddd081f6fe3af19c7d3c92a))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.53 ([6243a8d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6243a8d1e1ffe715a041bc78e3e312bf29061ff3))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.54 ([2dcb5cd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2dcb5cd78e13bfc310e04e9bcb279fbe46b7e531))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.55 ([fc2015e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fc2015e6a84e9728dc26f60162aba81588a9a606))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.56 ([de2f7f5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/de2f7f53437bfc5340e939ab395b310992e2a049))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.58 ([7027767](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/70277677b721ee951d868b6956c827395f2a5e0b))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.59 ([84da203](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/84da2035d413036da4dbac59e7753b834825a00a))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.60 ([bc52af0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bc52af0f8c735350be14c238e94134db1d744305))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.61 ([cf4420c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/cf4420c7358369fedba0edc61c8e49a25177a2f7))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.62 ([1cb8d3c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1cb8d3c2a8cc4505f95b771494c68ea47276f446))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.63 ([fe148cc](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fe148ccc6af1a397c0d105214262d50ed28d617e))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.64 ([c55884b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c55884bdc5c4c26c4d4b55d4e757f73a1c003fcb))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.65 ([5f76ca6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5f76ca6bf4db932c91e025c468fa7cf82dbe0ae9))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.66 ([e9205b8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e9205b810b1768c3a9be66b7f87268ff1d2edae5))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.67 ([2793283](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/279328372178f0444e4bb2303734f45daae211eb))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.68 ([5b8eafc](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5b8eafcf9b6fdd43de0aa80f9ea187eeb2c5e9b4))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.69 ([5f912df](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5f912dfe1c842580ecb32a57c3ef33c35477a8fd))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.70 ([8434e49](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8434e49952cf5c5f1a8514b171c0af84c723ee2d))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.71 ([2a1641c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2a1641c4633e32dfaa8fa3bf2efc03bc7706a5df))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.72 ([dfee3aa](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/dfee3aaf8fb1dc8cf5bc5334eaa78ade6dc7ea74))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.73 ([da43891](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/da43891b5808766db7e9739ec9d44f2125c0b6d5))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.74 ([6425637](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/64256376efe673f94d08d57170d3ec662bdeb35d))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.75 ([a23ac4c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a23ac4c4a63b51f87ba8339c9161d7a3761fc242))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.76 ([54a4350](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/54a4350366ec99ea1c03c16028f97ab6e7bdec74))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.77 ([11e280f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/11e280f0752dd25cdd18bfaee296488b2b00a0be))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.78 ([2fbafcd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2fbafcdae3e180b43fd0f3b431b60b143b7e5cc1))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.79 ([5f384e4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5f384e49831ebd6bb808a9f505615def06ce24c1))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.80 ([908ffa2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/908ffa2c9a32cb50a8316b274831c80cbb8a110e))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.81 ([b404680](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b404680774ae13a7ea910e031b8ec1ba25d99a25))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.82 ([6c4f6b6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6c4f6b631b64f28aa678216726b0ba07f519b18b))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.83 ([78081c3](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/78081c39ab40637384f61c9a03718899d143a7e4))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.84 ([c9a7b1b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c9a7b1b34fa4f8c48b4a054f69ca0f16637c1ba8))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.85 ([087038e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/087038e54d2c0a4ebd56e1b2d9b4ac577309f92c))
* **deps:** update dependency xarray to v2023.10.0 ([2f3519c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2f3519ca55e222ec6ba8dbda12336568033d1593))
* **deps:** update dependency xarray to v2023.10.1 ([6c5ba4b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6c5ba4b557ec652d0924603c332ab0d4077d34b2))
* **deps:** update dependency xarray to v2023.11.0 ([62a2df5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/62a2df500886ea51c881c2b2d8be76a72ab6ec34))
* **deps:** update dependency xarray to v2023.12.0 ([bd97af0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bd97af0b4f4b3d3ef727d55fb904e3e2d5c1cb05))
* **deps:** update dependency xarray to v2023.5.0 ([dc696b7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/dc696b77f9c3c4ac64555879ce1f83bbcd220c1d))
* **deps:** update dependency xarray to v2023.6.0 ([43428cf](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/43428cf18d7d5a3a0d0d1f47ed0e72113b82f4a9))
* **deps:** update dependency xarray to v2023.7.0 ([2723dfc](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2723dfca75342d0ddcc1b207e65ac87eba0af473))
* **deps:** update dependency xarray to v2023.8.0 ([399b063](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/399b0639753956f44bd5c4393a2ae18c4d612bfd))
* **deps:** update dependency xarray to v2023.9.0 ([25db928](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/25db92802b45e0b13502a890d7469d8de0c02eb6))
* **deps:** update dependency xarray to v2024 ([99cd5f3](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/99cd5f3530732c4dc1b4f41f3299368c191f1d64))
* **deps:** update dependency xarray to v2024.1.1 ([6727814](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/672781492400ee253ff0d5ea02dc12ecc7f14ae3))
* **deps:** update node.js to 18.17 ([af6e90b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/af6e90bcbfbf2de53faaee87fdef84e840a93f3b))
* **deps:** update node.js to 18.18 ([5f8e17f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5f8e17f97bc85cb1f06b0a50221a749d74178964))
* **deps:** update node.js to 20.10 ([21c0a8a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/21c0a8ac68efdd63b6294c35cb08ba72d9030553))
* **deps:** update node.js to 20.11 ([904e109](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/904e10968ed06699fce0f32ad8cf9a43e5749c50))
* **deps:** update node.js to 20.9 ([dc45ac1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/dc45ac1c6889250303ae43af0d0c69ab36d758d2))
* **deps:** update node.js to v20 ([b6fb42c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b6fb42c9969a260dec6cf0965324d5bca7c8f4c9))
* **deps:** update plugin com.gradle.enterprise to v3.13.2 ([10bd299](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/10bd299de7a566bd60f7d8dee09107d867d7896c))
* **deps:** update plugin com.gradle.enterprise to v3.13.3 ([a2ef3e9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a2ef3e98214b9152e41ae0aca6361ca18b5b9619))
* **deps:** update plugin com.gradle.enterprise to v3.13.4 ([9c8eca8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9c8eca87c26a1adf646980666cb869324704855b))
* **deps:** update plugin com.gradle.enterprise to v3.14 ([1802ebe](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1802ebe7013586e7c1e411233746eb3894fae509))
* **deps:** update plugin com.gradle.enterprise to v3.14.1 ([078910a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/078910af35fa6cd9376c814e55cacd9e31fd3cea))
* **deps:** update plugin com.gradle.enterprise to v3.15 ([fead052](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fead052ddf607e88da6bc26550bcc871b111705a))
* **deps:** update plugin com.gradle.enterprise to v3.15.1 ([1b4f634](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1b4f6344139f9e14be3735c07de296eb8b2f76dd))
* **deps:** update plugin com.gradle.enterprise to v3.16 ([bd94edd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bd94eddcc8966c3940a64b9304bffa2c064dfb50))
* **deps:** update plugin com.gradle.enterprise to v3.16.1 ([00228dd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/00228dd6027c76be82ddf72f2080d83085fc672b))
* **deps:** update plugin com.gradle.enterprise to v3.16.2 ([237d977](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/237d9777a8ef963003a32d4e511f5be2f3ef63c4))
* **deps:** update plugin gitsemver to v1.1.10 ([2179c2e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2179c2ed83ea64f35920984daf87e5b69e070a7f))
* **deps:** update plugin gitsemver to v1.1.11 ([9490d73](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9490d736389496f029a1c251efdfe50fe2ce62ff))
* **deps:** update plugin gitsemver to v1.1.13 ([cfa3bcd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/cfa3bcd00fe70b23907718efbfca0064c5a2cc14))
* **deps:** update plugin gitsemver to v1.1.14 ([b7e6a94](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b7e6a946ae634dd994c73b8d6f6eeaaee298acae))
* **deps:** update plugin gitsemver to v1.1.15 ([2586e3c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2586e3cccc388f374d6612aa9b279ca5ffcc1681))
* **deps:** update plugin gitsemver to v2 ([9981d23](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9981d2308942099e785b10b32cc488fad612e50b))
* **deps:** update plugin gitsemver to v2.0.1 ([2a6aac3](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2a6aac3f8ae1d204d48c1fd402f54a140988bd87))
* **deps:** update plugin gitsemver to v2.0.2 ([5fe6580](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5fe658098da02bd5b9f33f9c6377a36fa13a6e17))
* **deps:** update plugin gitsemver to v2.0.3 ([da7b9f1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/da7b9f1fe44f1e21d11c90b95abbac821c957d07))
* **deps:** update plugin gitsemver to v2.0.4 ([b35a491](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b35a4914c29b0ba70c3ec25e962d7ec36ec5bcb3))
* **deps:** update plugin gitsemver to v2.0.5 ([ec7d87d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ec7d87d689bf7ed66ee94181946fcfab0444aae5))
* **deps:** update plugin gitsemver to v3 ([950f217](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/950f2173b1be5da1c772e4d5bb6a72e7b9ef3b1e))
* **deps:** update plugin gitsemver to v3.1.0 ([7d87c19](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7d87c19ed976ce85c752450e8546d82f1a733923))
* **deps:** update plugin gitsemver to v3.1.1 ([144922b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/144922b63c23772562e0d19899eb3ab1cf00120f))
* **deps:** update plugin kotlin-qa to v0.42.0 ([35767d4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/35767d4cba5f451b015be7b4b955ed07975182bf))
* **deps:** update plugin kotlin-qa to v0.42.1 ([a2718ac](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a2718ac03ab436c902708b7f64ea6dff70a545d0))
* **deps:** update plugin kotlin-qa to v0.43.0 ([345d4da](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/345d4da04e3ac5177ee032cfcf0070016dedbba7))
* **deps:** update plugin kotlin-qa to v0.44.0 ([cc47d03](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/cc47d03e4c95d98672dc92999187bd63e9deee02))
* **deps:** update plugin kotlin-qa to v0.45.0 ([2cd8410](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2cd84105b9e02ae88c17f43ea87494b28be31d08))
* **deps:** update plugin kotlin-qa to v0.46.0 ([d642fa9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d642fa9a46a4ae6749aef6d4fe53026b1b559a14))
* **deps:** update plugin kotlin-qa to v0.46.1 ([462edb9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/462edb940305180324c5b1cf8f315a85515c44ca))
* **deps:** update plugin kotlin-qa to v0.47.0 ([99c6ae0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/99c6ae05c1c71dd33dc97e01e5ac2dac404a8397))
* **deps:** update plugin kotlin-qa to v0.47.1 ([3ff6789](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/3ff67896fe720468f0a01279c9c0eeff9565abac))
* **deps:** update plugin kotlin-qa to v0.48.0 ([a376e77](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a376e77edb62dce580ef4e56033a6a21ad34e0ef))
* **deps:** update plugin kotlin-qa to v0.49.0 ([cc73861](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/cc7386128ef46533b6bde98e39dc31bfee7e7e46))
* **deps:** update plugin kotlin-qa to v0.49.1 ([7a68b3a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7a68b3a54ff8178e18b27bdb90b2c37fdfefd626))
* **deps:** update plugin kotlin-qa to v0.50.0 ([aab3e98](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/aab3e980457c206467701f55a9b7522457931834))
* **deps:** update plugin kotlin-qa to v0.51.0 ([634ea66](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/634ea6651bd01fbe737843178cd658518d759859))
* **deps:** update plugin kotlin-qa to v0.51.1 ([0b52567](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/0b5256760ca8bfdd65c29e5b31e2dc189d3d76f5))
* **deps:** update plugin kotlin-qa to v0.52.0 ([07f92c5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/07f92c519206d0bb9c8bac5d64cecc95adc95b87))
* **deps:** update plugin kotlin-qa to v0.53.0 ([97be715](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/97be715c646a84762f3cfa16c585953a7a6184e4))
* **deps:** update plugin kotlin-qa to v0.54.0 ([a24c552](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a24c5523c025a9405b0e2d6c9848ebc7084a2cce))
* **deps:** update plugin kotlin-qa to v0.54.1 ([8be45c5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8be45c5f22c3bbb9459e9933031e9a83eb7d2e0c))
* **deps:** update plugin kotlin-qa to v0.55.1 ([d2e3dc9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d2e3dc9b0f168fb286b43202f20ea75a13878a30))
* **deps:** update plugin kotlin-qa to v0.55.2 ([2d16fbd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2d16fbd64c75cb7e6a6cb67ed9de706d239e447e))
* **deps:** update plugin kotlin-qa to v0.56.0 ([af45ef7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/af45ef7426e3c1ca700136acb203689db1cf2999))
* **deps:** update plugin kotlin-qa to v0.57.0 ([e990d56](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e990d56eb9a04702eb0b9adc761aa96088b4b1ff))
* **deps:** update plugin kotlin-qa to v0.57.1 ([033e0ed](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/033e0ed176fc075e63f66b7a8f3d8ace2c4c4e53))
* **deps:** update plugin kotlin-qa to v0.58.0 ([97d7e80](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/97d7e8066ed8e1a3d8ac7b72e6b0c8e1a5bcc0d3))
* **deps:** update plugin kotlin-qa to v0.59.0 ([869f788](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/869f788d89be2523515c261f73b0b5ac8568ea35))
* **deps:** update plugin kotlin-qa to v0.59.1 ([31703a8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/31703a88eb00501c70691c533ad80136197be064))
* **deps:** update plugin kotlin-qa to v0.60.0 ([94619a0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/94619a03c5c9548fe1ba1b3c9dc919c7cd968827))
* **deps:** update plugin multijvmtesting to v0.5.0 ([d0a918f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d0a918fae72da83d9bc571643cd9718b89a5a6bc))
* **deps:** update plugin multijvmtesting to v0.5.1 ([ea88f8f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ea88f8fb74d7c69e5bb956d2eeba83d4f5d54553))
* **deps:** update plugin multijvmtesting to v0.5.2 ([78f9255](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/78f925568c18360921040628fa45ec6c73ee3d54))
* **deps:** update plugin multijvmtesting to v0.5.3 ([5a99168](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5a991685faf011b0f19b5d085893ce5c227c957f))
* **deps:** update plugin multijvmtesting to v0.5.4 ([81f4ae7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/81f4ae73800f3f089ade6e8a23996331a3278d1c))
* **deps:** update plugin multijvmtesting to v0.5.5 ([d475b23](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d475b23169e5bf217c6e97ef85c8f45ffb31545a))
* **deps:** update plugin multijvmtesting to v0.5.6 ([c70be04](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c70be046a9d71cebebac7742076ecb0da781d707))
* **deps:** update plugin multijvmtesting to v0.5.7 ([da04983](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/da0498354a34daf28d6edcd6f67caf722dfbfe17))
* **deps:** update plugin multijvmtesting to v0.5.8 ([14293f6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/14293f6cf22244a103f29a2819677ff0546c27b4))
* **deps:** update plugin org.gradle.toolchains.foojay-resolver-convention to v0.6.0 ([5d4dd13](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5d4dd13971030546e434dd39954f8de4d2e4dcfc))
* **deps:** update plugin org.gradle.toolchains.foojay-resolver-convention to v0.7.0 ([3462ae2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/3462ae28ae09db143d94ee3b84c59b7c354f7f52))
* **deps:** update plugin org.gradle.toolchains.foojay-resolver-convention to v0.8.0 ([ec1c5d5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ec1c5d578128f5812308ec5afbf1f2a8e8bd71ed))
* **deps:** update python docker tag to v3.11.4 ([401d40a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/401d40a6034a0ecb43cf632c987224024c13f79c))
* **deps:** update python docker tag to v3.11.5 ([91462e5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/91462e5aee6102340ad7a53bc67830ded0100752))
* **deps:** update python docker tag to v3.12.0 ([3afabf0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/3afabf09103a78cd161a6c865daa347301ae7641))
* **deps:** update python docker tag to v3.12.1 ([cdc69c1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/cdc69c192b6512b86f84a617df4918966acd339a))
* **deps:** update python docker tag to v3.12.2 ([6902e77](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6902e77c809cd2e78a6f777e892ba7594f0311bb))


### Bug Fixes

* fix the launch of the graphics experiment ([9651958](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9651958bcfdc95cbf629944731210f512d335075))


### Build and continuous integration

* **deps:** update actions/[up/down]load-artifact action to v4 ([#668](https://github.com/DanySK/alchemist-experiments-bootstrap/issues/668)) ([e141441](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e141441c040fd0728e8496af77415bbe4edc7a36))
* **deps:** update actions/cache action to v4 ([734d6ee](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/734d6eea7dbc0a9be41f936fe54f4b8364c65ac2))
* **deps:** update actions/checkout action to v3.5.3 ([8398de8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8398de8c71e9fe129eeb00b3eb9ba856fc38d62b))
* **deps:** update actions/checkout action to v3.6.0 ([0fb9b54](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/0fb9b54a555fce11c0a9606fa08a3040657a5c2d))
* **deps:** update actions/checkout action to v4 ([7d232fc](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7d232fc3458ce438fdaa94d8de0499bf53bddc25))
* **deps:** update actions/checkout action to v4.1.0 ([7191901](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7191901ca2e66ece89ab3c7312ea0695ab02d424))
* **deps:** update actions/checkout action to v4.1.1 ([e578612](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e57861259751e3c381cdc388ad3ae4bcc55a7023))
* **deps:** update actions/download-artifact action to v4.1.1 ([58441a4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/58441a48414dbf6fe5eb02d5845aba2fa853f196))
* **deps:** update actions/download-artifact action to v4.1.2 ([468a700](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/468a7002c501820e7c0769e0669ef6d09882795a))
* **deps:** update actions/setup-node action to v3.7.0 ([4f3094b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4f3094bcd995fcba2289f70d41ee0a36e20625a6))
* **deps:** update actions/setup-node action to v3.8.0 ([a8f0852](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a8f08528dbbe41f5a9d5179a2e80187e36c5cc86))
* **deps:** update actions/setup-node action to v3.8.1 ([ad2951e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ad2951e373c1056eff3f7ecbbdeb9702a5382788))
* **deps:** update actions/setup-node action to v3.8.2 ([5512df3](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5512df32c1cd1298545a652a5d6f10a0d4bee909))
* **deps:** update actions/setup-node action to v4 ([1ac1c22](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1ac1c229789a88b538b32b4887fb98c11b241dc4))
* **deps:** update actions/setup-node action to v4.0.1 ([ff53f47](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ff53f4791275e914e9aa395c5dfb9213943e95e9))
* **deps:** update actions/setup-node action to v4.0.2 ([b98efe7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b98efe76a2aef820eaa1a4ef7c964c19fb13a1f2))
* **deps:** update actions/setup-python action to v4.6.1 ([8810c2c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8810c2ce7e4e0d5846b7666594d0d7984fcde1a2))
* **deps:** update actions/setup-python action to v4.7.0 ([58323d3](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/58323d32f044725fffc79e6d25bbf99bebe22690))
* **deps:** update actions/setup-python action to v4.7.1 ([e4bb109](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e4bb109eb0d74fc2f39bc586d89531c05e32f17f))
* **deps:** update actions/setup-python action to v4.8.0 ([d302a93](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d302a93e07dbc98174b77a542772200e025bd7c1))
* **deps:** update actions/setup-python action to v5 ([23c7548](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/23c7548d3c7fc25c7ac3709a5fef627db651c991))
* **deps:** update actions/upload-artifact action to v3.1.3 ([31d9cd5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/31d9cd57b221e416ccc3dff30f48a57dd2048488))
* **deps:** update actions/upload-artifact action to v4.1.0 ([00cf691](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/00cf691156297e13b5ea64b40e2da6faf6bbdaaa))
* **deps:** update actions/upload-artifact action to v4.2.0 ([c22ade2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c22ade2153f4bc29d600a09c781647e018a292bb))
* **deps:** update actions/upload-artifact action to v4.3.0 ([5b408be](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5b408bec09eb7290e1f2c4d1774ade128cc1b365))
* **deps:** update actions/upload-artifact action to v4.3.1 ([cff6bed](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/cff6bed3bcee8b1d95962365efa27db5000d1a84))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.10 ([d846076](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d846076d24ac2f50be15db10b043c6c2a1a3d30d))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.11 ([d6ea2fd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d6ea2fd65571320c5fa3fafcccfebd81e9d99df9))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.12 ([fbe69a2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fbe69a2888206b31afa0cd3a389d85b2d3f164c9))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.13 ([2a98f7b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2a98f7bf33217da4b50fa933c0113c2e1f68735e))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.14 ([a1ea117](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a1ea11705d99d1f982e7509f2fdeedfb66f5fd30))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.15 ([07347f6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/07347f69ec9eff374da3e6f32d73aa5ba5a829b2))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.16 ([27d6c15](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/27d6c1543466cfb14f00e6f15ca2b0e293b9ad2f))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.17 ([228dac7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/228dac7aa1c1e60f1f4f85c79118b1199bd8530e))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.19 ([fa4ddaf](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fa4ddaf1862d7a866fbae9e8ec3a4e7053097c73))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.20 ([5dd34ab](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5dd34ab62db14b3733cafff7c6293459333c62cf))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.5 ([05f42f4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/05f42f47794ccb77c72aef30b44c9e267a7b2874))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.6 ([1997e05](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1997e05cb269448540c387f05f369b8938fcd913))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.7 ([00983e7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/00983e71d0df0c3ff5c19ec97817345703819f57))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.8 ([6d350ab](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6d350ab66d925c5d57be3e92d799224adc680579))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.9 ([b791675](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b7916755efc22baa88e6a39bd7d5ef82f2baae15))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.3.0 ([f7ecf8b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f7ecf8b9ba94ee0c997c3fcf78d351d521243636))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.3.1 ([87d5e18](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/87d5e181b5494d7cf4ccef8a44e080fb022e9476))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.3.2 ([4e09141](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4e091419fe9ee21135fce9fa1f854fcf69d43660))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.4.0 ([a999818](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a9998188207f33cdb0e1112df5101e7019d955ff))
* **mergify:** use the shared config ([e9bb4fe](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e9bb4fe72eefa42fe25016785438069e22907f70))


### General maintenance

* **release:** update gradle.properties .env versions to 1.2.0 [skip ci] ([a772834](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a7728345edc11cba3a469fd7ab68145bf19cc85d))

## [1.2.0](https://github.com/DanySK/alchemist-experiments-bootstrap/compare/1.1.3...1.2.0) (2023-05-05)


### Features

* prefer fill_between to express error in line chart ([#479](https://github.com/DanySK/alchemist-experiments-bootstrap/issues/479)) ([43dc387](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/43dc387c6b9a07f6382d109dac3566ed9447224f))


### General maintenance

* **release:** update gradle.properties .env versions to 1.1.3 [skip ci] ([64691b2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/64691b2ecab41a7316b8ccac63d6a9db8ae9f381))


### Build and continuous integration

* apply org.gradle.toolchains.foojay-resolver-convention ([#407](https://github.com/DanySK/alchemist-experiments-bootstrap/issues/407)) ([fdf0a1d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fdf0a1d8bba8eeb95258d23d2235277394f65822))
* **deps:** update actions/checkout action to v3.5.1 ([cf0d0f5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/cf0d0f55fbe6e81fadc34869485aa0a2f255f566))
* **deps:** update actions/checkout action to v3.5.2 ([3ba88c7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/3ba88c71a2cad4bd9ee3862c61157f76d689f3cf))
* **deps:** update actions/setup-python action to v4.6.0 ([ecbab42](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ecbab4273876dc0e63e8639babd35aadeef9965b))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.22 ([060d51d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/060d51d33710d6723bf2ce72f7820d20322c43fa))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.23 ([98e1edd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/98e1edde67bd47f35e38d8d737db52a9a206775b))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.24 ([ba72398](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ba72398ab44cec95c176280cdc82e59cc1f6dc39))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.25 ([7cfab9d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7cfab9d04e683d172eb29c753af8785e66f5613f))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.26 ([15afea2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/15afea22abf771a1f1cccdafb2feb54aa1067dde))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.0 ([e5a37d8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e5a37d8caed6eae4b6a1b9750ae15fc1d905a9e3))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.3 ([ecd00b7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ecd00b724905f50d5f3032246187d88e756360e6))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.2.4 ([832f27a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/832f27abbba701a51148d68703319462b8c9facf))


### Dependency updates

* **deps:** update alchemist to v25.10.0 ([f3908ef](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f3908ef737b55d61c1fd01fd926d1996ae51bc9c))
* **deps:** update alchemist to v25.11.0 ([8f91f4e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8f91f4e3bf4dc03c8c69de3f202001dd85530727))
* **deps:** update alchemist to v25.12.0 ([35283d4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/35283d4eaa911c5f3bfc71176a9eb424a6928093))
* **deps:** update alchemist to v25.13.0 ([9e0e5e3](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9e0e5e349b1046b4e550ef74e6caaea0c70a5fdb))
* **deps:** update alchemist to v25.13.1 ([92999a8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/92999a8bfb8fd711d1a7126d9922f6eab26dadfc))
* **deps:** update alchemist to v25.14.0 ([1b47a92](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1b47a92cc0ee0b11621caf876ccb9fe2d056b009))
* **deps:** update alchemist to v25.14.2 ([e087470](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e0874704503fe78687f312a4da289a629a3ab75d))
* **deps:** update alchemist to v25.14.3 ([d6e1d53](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d6e1d531eed94e062de25b41d6120d490a7eece8))
* **deps:** update alchemist to v25.14.4 ([bef5e74](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bef5e74661858c1d520692da10993a4c2bdbadac))
* **deps:** update alchemist to v25.14.5 ([dd65db2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/dd65db23072db3a283f37a0e20cc4ddf71c185e9))
* **deps:** update alchemist to v25.14.6 ([72decad](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/72decad1db4af48ddf05075720a29a1fd485ea4b))
* **deps:** update alchemist to v25.15.2 ([9acc552](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9acc5522804c0061e734c6624a17ac068bd59b04))
* **deps:** update alchemist to v25.15.3 ([e756ead](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e756eadd17d60c472c304a1d9f43b123215b9efe))
* **deps:** update alchemist to v25.15.4 ([16950dd](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/16950dd8fab38358f78945e4efd10a806efb269e))
* **deps:** update alchemist to v25.15.5 ([9c368dc](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9c368dca29fde80d4a9057ae0eb2db66159ac44e))
* **deps:** update alchemist to v25.16.0 ([39176ac](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/39176ac1d0a17fbc3e6aafa7301c4139cb45605d))
* **deps:** update alchemist to v25.16.1 ([d7ffc14](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d7ffc1419f5ebf415fba8a1973b8205fc35af465))
* **deps:** update alchemist to v25.8.0 ([207316a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/207316a8457a83a0be50786abc0326fbd2a3fb03))
* **deps:** update alchemist to v25.9.0 ([f141b57](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f141b57a4f4983c7e22422cd7397ea67445a7fcf))
* **deps:** update alchemist to v25.9.1 ([bd5f61a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bd5f61a5a2fb549e487ad5337e7cbffc5324e9b4))
* **deps:** update alchemist to v25.9.2 ([ec6c71c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ec6c71c00134dbf4c35e1518ea250d40eb8ea201))
* **deps:** update alchemist to v25.9.3 ([ec1e1ee](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ec1e1ee09a4b2236a91a158f473957d9891aef9f))
* **deps:** update alpine docker tag to v3.17.2 ([1112807](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/1112807129433d69997588767cd22e3bd7f21ddd))
* **deps:** update alpine docker tag to v3.17.3 ([8a0af30](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8a0af30aeea498306cc7c9f15f1e4300f4522815))
* **deps:** update dependency gradle to v8 ([8cbddea](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8cbddea867aa5d63e4107783e96e18ed87008c65))
* **deps:** update dependency gradle to v8.0.2 ([e6f90eb](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e6f90eb54aacfadb99e128bf6fe8afadf81f1b16))
* **deps:** update dependency gradle to v8.1 ([b49d8d5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b49d8d5dc5c16403b6e8d30e7800e0e840b26225))
* **deps:** update dependency gradle to v8.1.1 ([57af727](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/57af7275a8c6c305f8945102847660ef20880253))
* **deps:** update dependency matplotlib to v3.7.0 ([fc71450](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fc7145080c8aa6eba29c22ffaaf937c4070dc64e))
* **deps:** update dependency matplotlib to v3.7.1 ([58d8bc6](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/58d8bc6b6bea9d85f6ba14f91f69a78a9fb754b9))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.8.10 ([4ee4c7c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4ee4c7c59790d05a9bc213fbfa17f2a8378ebc7d))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.8.20 ([6a8a5e5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6a8a5e510a1b74400b8c6de6b46bcf49c4b80ed1))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.8.21 ([b9268b5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b9268b5fe6bb7161c7a61dc3165798fcb0843b1a))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.23 ([9749694](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9749694abd96e7a6a882e6026466f9bd84f569b4))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.24 ([fc417e2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fc417e2ebc41fa531a7fa251d0c12261e49fb885))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.25 ([300065a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/300065af5c22bdabfc469141713f33ddc412b5d8))
* **deps:** update dependency semantic-release-preconfigured-conventional-commits to v1.1.26 ([9a6248c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/9a6248c3bc08e719a86f3c9f5a18b0494ae3c008))
* **deps:** update dependency xarray to v2023.2.0 ([64febd8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/64febd82eac86f365dca083e9dfabf88cab7ddce))
* **deps:** update dependency xarray to v2023.3.0 ([5866720](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/58667209784d5bdb06858710522cf16f07eeaa67))
* **deps:** update dependency xarray to v2023.4.0 ([bfe7dcb](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/bfe7dcb7a58c7e344909a2ce84c59995b0f0fbbd))
* **deps:** update dependency xarray to v2023.4.1 ([ba80193](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ba8019369cbea2f1693582a3699dcb1db3c54b29))
* **deps:** update dependency xarray to v2023.4.2 ([704c912](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/704c9128c79d280db37a2b51a352b5fd874306c5))
* **deps:** update node.js to 18.14 ([3658e0f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/3658e0fccbc4b9231481e1882381d88a85b98e5d))
* **deps:** update node.js to 18.15 ([262c0af](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/262c0af7f7f89e39c149c8a2d077724fb4281db6))
* **deps:** update node.js to 18.16 ([2222cbe](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/2222cbee7cb3fa0c1c5b10b8a3b65a30fd910a97))
* **deps:** update plugin com.gradle.enterprise to v3.12.3 ([6ccf8c9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6ccf8c919a226b280560b9b74d4f306765a9cc09))
* **deps:** update plugin com.gradle.enterprise to v3.12.4 ([126b903](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/126b90374651296f67669222df0ca755612abea9))
* **deps:** update plugin com.gradle.enterprise to v3.12.6 ([8b30cc1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8b30cc118900e8cfa7a5f7eb2c8a6773b241e429))
* **deps:** update plugin com.gradle.enterprise to v3.13 ([7464c3c](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7464c3c60e88d6687858470dfe8134c6e471fd0b))
* **deps:** update plugin com.gradle.enterprise to v3.13.1 ([a947cd5](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a947cd5e5b18b15971d8d6280d072aadd9c5d6f6))
* **deps:** update plugin gitsemver to v1.0.2 ([6582785](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/65827853038abc5a4eba6dcd8fe2427dfe99d68b))
* **deps:** update plugin gitsemver to v1.1.1 ([6a1da20](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6a1da20a52d2a05de3c30f46e845c3e1cc77fbdf))
* **deps:** update plugin gitsemver to v1.1.2 ([b77f3a9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b77f3a9fd6cd3131af7d890f7dc5e96e2522db1a))
* **deps:** update plugin gitsemver to v1.1.3 ([c8aa3ab](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c8aa3abcf7d36db096c4bc39ad058803f1809022))
* **deps:** update plugin gitsemver to v1.1.4 ([e62f9a8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e62f9a8d406c2253b8841c1f8b19f7eb51834151))
* **deps:** update plugin gitsemver to v1.1.6 ([126bebe](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/126bebe1fcd28441d0d610094523d07368ff47c6))
* **deps:** update plugin gitsemver to v1.1.7 ([7c69407](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7c694070bbba370e523cdec8a7ed8cf1f075e640))
* **deps:** update plugin gitsemver to v1.1.8 ([c64aec7](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c64aec766f1865e51a499bcebdea3b793f578d89))
* **deps:** update plugin gitsemver to v1.1.9 ([e9905da](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e9905dae7f3e334a00879243985ad2009a95faa4))
* **deps:** update plugin kotlin-qa to v0.34.0 ([c02e443](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c02e4434498ee8bc103bc88107d698f66c82af74))
* **deps:** update plugin kotlin-qa to v0.34.1 ([63d0e84](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/63d0e84fdcbc6a1d8ca2c7083811e7149a8a8f69))
* **deps:** update plugin kotlin-qa to v0.34.2 ([b06035e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/b06035efcc8f8ec7e86f1430e2bc893ba586d8c1))
* **deps:** update plugin kotlin-qa to v0.35.0 ([41031a1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/41031a1985ab9f3333fb4d981822663d641bc8e7))
* **deps:** update plugin kotlin-qa to v0.36.1 ([f4ff837](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f4ff837a0e5b5ef3321d69f1373847783e551ebe))
* **deps:** update plugin kotlin-qa to v0.38.1 ([a290e6f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a290e6f03d496dd90521707a49cdc6e24136e777))
* **deps:** update plugin kotlin-qa to v0.38.2 ([fd1cb22](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/fd1cb2281291979529741a06139dc052fcffb8b2))
* **deps:** update plugin kotlin-qa to v0.39.0 ([db65d17](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/db65d17ee17afb3834436a77b625e19b8359ffbe))
* **deps:** update plugin kotlin-qa to v0.40.0 ([7b9b396](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7b9b396b2dd96e9633f88fb3f5f25e02e6e212f8))
* **deps:** update plugin kotlin-qa to v0.41.0 ([ed52452](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ed52452b45d6d212dda1a1878a68249d76c57208))
* **deps:** update plugin multijvmtesting to v0.4.16 ([81f839d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/81f839d5baf9239784a0b7940e4551aa0ac2af61))
* **deps:** update plugin multijvmtesting to v0.4.17 ([39206e0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/39206e0ae87e9898718efb2d60cbeb420ba68f45))
* **deps:** update plugin multijvmtesting to v0.4.19 ([5e9eb9f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/5e9eb9fd51d3461d3e2826a937c309abeb4df8b4))
* **deps:** update plugin multijvmtesting to v0.4.20 ([62fd0f9](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/62fd0f961261880b7501d5896044a503fddf252a))
* **deps:** update plugin multijvmtesting to v0.4.21 ([87fe715](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/87fe7159ed4194e5f3a04b7c09d9c3a62506c3ea))
* **deps:** update plugin multijvmtesting to v0.4.22 ([7f1a26e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/7f1a26e20b0b4d152e04ad664169296ca80bdecb))
* **deps:** update plugin multijvmtesting to v0.4.23 ([4bf5b18](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4bf5b18293198fddfdbdce66dd580781315e9c23))
* **deps:** update plugin org.gradle.toolchains.foojay-resolver-convention to v0.5.0 ([de5d2bb](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/de5d2bbe901c34505aae01292b0143ebea4b883e))
* **deps:** update python docker tag to v3.11.2 ([3317d68](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/3317d68d728bbc45ab13670806ea629c81095079))
* **deps:** update python docker tag to v3.11.3 ([a6d8861](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a6d886122cab5bd922b912fd9612ee6996f60f47))

## [1.1.3](https://github.com/DanySK/alchemist-experiments-bootstrap/compare/1.1.2...1.1.3) (2023-01-29)


### Bug Fixes

* **release:** move pushes with refreshed versions to the publish phase ([c185161](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/c1851619d8e302165c6eaeaf66ae52c561e1a54c))


### Documentation

* **readme:** add a way to re-run the charts generation without re-running experiments ([56c8965](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/56c8965fa8c8f9cff91e99455056392386e37440))
* **readme:** prepare a readme stub ([e7d7432](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/e7d743218d65a27dad7c463ca2b0c2f83d07ecc0))


### Dependency updates

* **deps:** update python docker tag to v3.11.1 ([77f15c1](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/77f15c137c30bd23c948c17e32ec708ce7d88e3c))


### Build and continuous integration

* always rebuild the container images ([6a1037a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6a1037a5a902e85093f4c2d8a22c7adde31bce53))
* validate the Gradle wrapper on Linux only ([8846942](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8846942331bc06bec28d440b42be31054bf07a33))

## [1.1.2](https://github.com/DanySK/alchemist-experiments-bootstrap/compare/1.1.1...1.1.2) (2023-01-24)


### Bug Fixes

* **release:** add a preparatory commit for the version updates ([f811f3d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f811f3d876b776baf6529dddd36f37cada4236b9))
* **release:** push the updated files after version calculation ([0311e56](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/0311e56a5110d883c2c71a8a7002445fdbf9adf6))


### General maintenance

* **release:** update version descriptors to 1.1.2 ([298c888](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/298c88848084eb3b9af658e867d736bec4e01496))


### Build and continuous integration

* **release:** skip the CI process for release commits ([8a0732f](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/8a0732f82b17f5dc85382c4de2716f4d6a10f486))

## [1.1.2](https://github.com/DanySK/alchemist-experiments-bootstrap/compare/1.1.1...1.1.2) (2023-01-24)


### Bug Fixes

* **release:** add a preparatory commit for the version updates ([f811f3d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/f811f3d876b776baf6529dddd36f37cada4236b9))
* **release:** push the updated files after version calculation ([0311e56](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/0311e56a5110d883c2c71a8a7002445fdbf9adf6))

## [1.1.1](https://github.com/DanySK/alchemist-experiments-bootstrap/compare/1.1.0...1.1.1) (2023-01-24)


### Bug Fixes

* **release:** move the preparation commands to the prepare phase ([ee52de2](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ee52de20fa26653febcfdc158884b38ad2a209d8))

## [1.1.0](https://github.com/DanySK/alchemist-experiments-bootstrap/compare/1.0.1...1.1.0) (2023-01-24)


### Features

* run simulations with more variables to make decent charts ([77e6d6d](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/77e6d6d64b2ce74992c2c9cd35b75a2860571a50))


### Bug Fixes

* **build:** remove deprecated variable assignment ([6099baf](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/6099baf6f9e666e066f8fd7623a6198521eb46b4))
* **build:** remove generated data after the simulation container build ([a79dab4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a79dab48e41dfc09b213afcae7e29f9767fb9079))
* **build:** switch to modern JSON effects ([a0ede3b](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a0ede3bb2bfa5a868d3d918e2b59d2edd95dd39c))
* **ci:** read the docker password from the CI secrets ([689a3a4](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/689a3a40fef2e65a9dc6eaab3a07378af955b510))
* **ci:** use a separate concurrency group for the container check job ([d06f389](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d06f3894edb14f67693aa8da043f670cdbb4a2b3))
* **release:** publish the images built in CI ([4c7d849](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/4c7d8495adfac26818aa87cab08cc25f954d59aa))


### Dependency updates

* **deps:** update alchemist to v25.7.2 ([00b615a](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/00b615ae14b62f677b7410886fef189a61c9645c))
* **deps:** update dependency xarray to v2023 ([32cc855](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/32cc85583d6967e1d2e740a9641c5011997bb979))


### Build and continuous integration

* add a docker check job ([ca6ac93](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/ca6ac934f5e3aeb63eb3350a2d4bdc22ea7cb056))
* prepare a containerized version of the experiment ([d0ff60e](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d0ff60eec241b67a43064796fdc24f930d698394))


### General maintenance

* **build:** remove aggressive heap settings ([a40d0be](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/a40d0be30b95ec93b5e8baaa32961211d5d6c706))
* delete Eclipse settings ([807f8d0](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/807f8d0835162d4bf96baf81b4dd686c60abbd67))
* ignore idea, spyder, gradle, and charts folder ([d92b6e8](https://github.com/DanySK/alchemist-experiments-bootstrap/commit/d92b6e8d52c9cfe0d83de9dc5a03cf26e93d18e6))

## [1.0.1](https://github.com/DanySK/Alchemist-experiments-startup/compare/1.0.0...1.0.1) (2023-01-19)


### Documentation

* clarify the need for the DEPLOYMENT_TOKEN in the readme ([534533b](https://github.com/DanySK/Alchemist-experiments-startup/commit/534533b7826dce6ebf3304fdea2807aa1c9d982b))


### Build and continuous integration

* cancel in-progress charts rendering on new pushes ([a377004](https://github.com/DanySK/Alchemist-experiments-startup/commit/a3770040cdc89b8a3805961d5774f8bddeb2c7b7))
* cancel in-progress simulation checks on new pushes ([0e37030](https://github.com/DanySK/Alchemist-experiments-startup/commit/0e37030dae0c91f615b5fa2c3bedf2d18aca2194))
* **deps:** update dependency macos github actions runner to v12 ([e219ab1](https://github.com/DanySK/Alchemist-experiments-startup/commit/e219ab16b0ceb4c8ca75795660c5d3b4fba9a3f9))
* **deps:** update dependency windows github actions runner to v22 ([238a30d](https://github.com/DanySK/Alchemist-experiments-startup/commit/238a30dda649405d7a5e3fd8729d8182df499d4d))
* limit the concurrency of the release jobs ([6b3c238](https://github.com/DanySK/Alchemist-experiments-startup/commit/6b3c238de08c2f9b9501c21f75ecb2001f0294a8))
* recover the build functionality ([6197c5c](https://github.com/DanySK/Alchemist-experiments-startup/commit/6197c5cc5ecd23272d7dbe82765a3a5b8c06d419))


### Dependency updates

* **deps:** update alchemist to v25.7.1 ([b54e7c5](https://github.com/DanySK/Alchemist-experiments-startup/commit/b54e7c5e7930c35772d55ebb19c2c163f5e85aec))
* **deps:** update plugin com.gradle.enterprise to v3.12.2 ([4513d0f](https://github.com/DanySK/Alchemist-experiments-startup/commit/4513d0feb288272453a4ff1b9643bfa802f907ac))
* **deps:** update plugin gitsemver to v1 ([87b688e](https://github.com/DanySK/Alchemist-experiments-startup/commit/87b688ee5c2fe3072f8f3107686d25aa6f6749a2))
* **deps:** update plugin tasktree to v2.1.1 ([761aded](https://github.com/DanySK/Alchemist-experiments-startup/commit/761aded54c3f9da7f35e1a035920b859d79d6bbf))
* **deps:** update python docker tag to v3.11.1 ([920a338](https://github.com/DanySK/Alchemist-experiments-startup/commit/920a338227325a047a8d5ab309e6f92390168e1a))

## 1.0.0 (2023-01-19)


### Bug Fixes

* **charting:** convert Java's 'Infinity' to a Python-friendly 1e30000 ([f432fae](https://github.com/DanySK/Alchemist-experiments-startup/commit/f432fae470f7bc6ade3c572d3e794acc620059c3))
* **deps:** update alchemist to v24.1.2 ([a39e289](https://github.com/DanySK/Alchemist-experiments-startup/commit/a39e28918a9bfdcf2bdbc5c95c4def2ca9baf7a7))


### General maintenance

* delete unused scripts folder ([d20f7c5](https://github.com/DanySK/Alchemist-experiments-startup/commit/d20f7c56a30bd5e6e8e8faa8b00542ce128b616e))
* delete unused version.properties ([068028c](https://github.com/DanySK/Alchemist-experiments-startup/commit/068028c2e3679ce47a100bdbc1750509e8da3a3a))


### Dependency updates

* **deps:** update alchemist to v18 ([7620fbd](https://github.com/DanySK/Alchemist-experiments-startup/commit/7620fbd05268f2b0bc8e678e017ab86b90c55389))
* **deps:** update alchemist to v19 ([8a6073f](https://github.com/DanySK/Alchemist-experiments-startup/commit/8a6073f51376617aaf2b5ff2547ec31125048e2e))
* **deps:** update alchemist to v19.0.1 ([34d2fbe](https://github.com/DanySK/Alchemist-experiments-startup/commit/34d2fbe1af969fe05204f1be5a4859764c8534d7))
* **deps:** update alchemist to v19.0.2 ([cefb051](https://github.com/DanySK/Alchemist-experiments-startup/commit/cefb0510c26df3ec20a63f5c368adcc5abc67876))
* **deps:** update alchemist to v19.0.3 ([88da774](https://github.com/DanySK/Alchemist-experiments-startup/commit/88da7743c4dc58ecb471a245ccf75d030b159ae2))
* **deps:** update alchemist to v19.0.4 ([e7a2544](https://github.com/DanySK/Alchemist-experiments-startup/commit/e7a2544dcf6459594ff64cae09dce511776f907e))
* **deps:** update alchemist to v19.1.0 ([5614547](https://github.com/DanySK/Alchemist-experiments-startup/commit/56145479ccd56fcec4f72e3852ae5edcaa8e7c13))
* **deps:** update alchemist to v19.1.1 ([9fa0a33](https://github.com/DanySK/Alchemist-experiments-startup/commit/9fa0a339b787a927e3c0fcd48e86a47fb690f578))
* **deps:** update alchemist to v19.2.0 ([8bf0669](https://github.com/DanySK/Alchemist-experiments-startup/commit/8bf0669c08db14e75072605faa09de4596e78168))
* **deps:** update alchemist to v20 ([aa28f14](https://github.com/DanySK/Alchemist-experiments-startup/commit/aa28f146f6320da547e8de526d498190ef3e2e51))
* **deps:** update alchemist to v20.0.1 ([4d2ee2c](https://github.com/DanySK/Alchemist-experiments-startup/commit/4d2ee2cc3e16c08edae585cfda0cef831b4b631d))
* **deps:** update alchemist to v21 ([f2e52c2](https://github.com/DanySK/Alchemist-experiments-startup/commit/f2e52c258b29acebc339189fbef720c64b09eaf7))
* **deps:** update alchemist to v21.0.1 ([a46aaae](https://github.com/DanySK/Alchemist-experiments-startup/commit/a46aaae1d7c6fbbcea1e547440c7a50c4c6cf71e))
* **deps:** update alchemist to v21.0.8 ([7ab2a39](https://github.com/DanySK/Alchemist-experiments-startup/commit/7ab2a397b1216267e3d3552cae91f9b6be8b1a8a))
* **deps:** update alchemist to v22 ([ed65e60](https://github.com/DanySK/Alchemist-experiments-startup/commit/ed65e607777043cc457b4508e6e62434e09b327d))
* **deps:** update alchemist to v23 ([2c10f27](https://github.com/DanySK/Alchemist-experiments-startup/commit/2c10f2730bbeb827344d16943f56ea1a83175e44))
* **deps:** update alchemist to v24 ([2114e88](https://github.com/DanySK/Alchemist-experiments-startup/commit/2114e886b9f6621ec9516a674fd27f15b6553a1a))
* **deps:** update alchemist to v24.0.3 ([85abe2e](https://github.com/DanySK/Alchemist-experiments-startup/commit/85abe2eb86ccdd205b14059f7a9b5044a2116c63))
* **deps:** update alchemist to v24.0.4 ([3ecee08](https://github.com/DanySK/Alchemist-experiments-startup/commit/3ecee08b71f5e12864de0b50a37d4152887a7dcf))
* **deps:** update alchemist to v24.0.5 ([6ba51e2](https://github.com/DanySK/Alchemist-experiments-startup/commit/6ba51e2b9045b3e58e92b2b022764f2f3f67884b))
* **deps:** update alchemist to v24.0.7 ([71c2035](https://github.com/DanySK/Alchemist-experiments-startup/commit/71c2035ad3d302ec225911de30defe42e41cfefa))
* **deps:** update alchemist to v24.0.8 ([2bdf0d2](https://github.com/DanySK/Alchemist-experiments-startup/commit/2bdf0d2637b61ac8c7948561ff65da17df75bfd3))
* **deps:** update alchemist to v24.1.0 ([ee1726b](https://github.com/DanySK/Alchemist-experiments-startup/commit/ee1726bfd66e8031ba0a23b0f957445c112c849b))
* **deps:** update alchemist to v24.1.1 ([5d7702e](https://github.com/DanySK/Alchemist-experiments-startup/commit/5d7702e6e2aa4bb445954407a5f8c2b37ddb039f))
* **deps:** update alchemist to v24.1.10 ([62b199d](https://github.com/DanySK/Alchemist-experiments-startup/commit/62b199da2af78bf2cb75bce8f1e6cc539ca17f15))
* **deps:** update alchemist to v24.1.11 ([96c5277](https://github.com/DanySK/Alchemist-experiments-startup/commit/96c5277a8b080ae5b8c887b9410eeb4771610095))
* **deps:** update alchemist to v24.1.12 ([ac4e62e](https://github.com/DanySK/Alchemist-experiments-startup/commit/ac4e62e1988bb220f63411834ef330f6699ca01b))
* **deps:** update alchemist to v24.1.13 ([f5c0199](https://github.com/DanySK/Alchemist-experiments-startup/commit/f5c0199016bf02ce482fcf20596256758fac0140))
* **deps:** update alchemist to v24.1.14 ([fc92d67](https://github.com/DanySK/Alchemist-experiments-startup/commit/fc92d6708f8bc8e628f044af59dace2fb6995d09))
* **deps:** update alchemist to v24.1.15 ([16c3926](https://github.com/DanySK/Alchemist-experiments-startup/commit/16c3926759d394155e16c1028f77108199268feb))
* **deps:** update alchemist to v24.1.16 ([a46b725](https://github.com/DanySK/Alchemist-experiments-startup/commit/a46b7259e90967517ae59b939b0e89ba5beecfb1))
* **deps:** update alchemist to v24.1.3 ([c568d69](https://github.com/DanySK/Alchemist-experiments-startup/commit/c568d694057cfe2ea910b8c8c4cf96e2a51aa3a7))
* **deps:** update alchemist to v24.1.4 ([b8e0ccd](https://github.com/DanySK/Alchemist-experiments-startup/commit/b8e0ccdb0b9fee66855aeb70cdc73e4cee4b2571))
* **deps:** update alchemist to v24.1.5 ([7878a29](https://github.com/DanySK/Alchemist-experiments-startup/commit/7878a299014d19b27616a63d14a81538a74d5a61))
* **deps:** update alchemist to v24.1.6 ([d8a385c](https://github.com/DanySK/Alchemist-experiments-startup/commit/d8a385cd5a214dd7439163999a338e2c96e84e65))
* **deps:** update alchemist to v24.1.7 ([37f4d79](https://github.com/DanySK/Alchemist-experiments-startup/commit/37f4d79a245970988e15fc6deda95102638e75fe))
* **deps:** update alchemist to v24.1.9 ([f7984a9](https://github.com/DanySK/Alchemist-experiments-startup/commit/f7984a960d3bbf5906d10474cffe4dac0657d673))
* **deps:** update alchemist to v25 ([7db2f48](https://github.com/DanySK/Alchemist-experiments-startup/commit/7db2f48b4f4842511221efeb9eeab977e49f8294))
* **deps:** update alchemist to v25.2.0 ([880b5b9](https://github.com/DanySK/Alchemist-experiments-startup/commit/880b5b9aac081194bddf469a27cf480a99898285))
* **deps:** update alchemist to v25.2.1 ([5b7309e](https://github.com/DanySK/Alchemist-experiments-startup/commit/5b7309e3beb4c4544d61597ef68df2748572f32a))
* **deps:** update alchemist to v25.3.0 ([2d6fab5](https://github.com/DanySK/Alchemist-experiments-startup/commit/2d6fab591f2f55f447650a9d49dae2e3a8f362f1))
* **deps:** update alchemist to v25.4.0 ([6ac914a](https://github.com/DanySK/Alchemist-experiments-startup/commit/6ac914a735e1ed0edb06f3e957ccba54dd99593b))
* **deps:** update alchemist to v25.4.1 ([0e9be1f](https://github.com/DanySK/Alchemist-experiments-startup/commit/0e9be1fbe96420e43f844d8c7ebab6a44ac488aa))
* **deps:** update alchemist to v25.4.2 ([fe0b221](https://github.com/DanySK/Alchemist-experiments-startup/commit/fe0b2211a9ded5b05baf3a7791731fa9c4a1cbf1))
* **deps:** update alchemist to v25.5.0 ([71da78f](https://github.com/DanySK/Alchemist-experiments-startup/commit/71da78fb38d07a46c2b72939507cebf4295e5ce9))
* **deps:** update alchemist to v25.6.0 ([7ec8059](https://github.com/DanySK/Alchemist-experiments-startup/commit/7ec805978ff925f3622007e26870c3b08f2fd603))
* **deps:** update alchemist to v25.7.0 ([6cdf812](https://github.com/DanySK/Alchemist-experiments-startup/commit/6cdf812d480927c85447f260a2809d5dc8c71703))
* **deps:** update dependency gradle to v7.3 ([f3627dd](https://github.com/DanySK/Alchemist-experiments-startup/commit/f3627ddb0809bfb9fc12e8e6de993ebaccdd704d))
* **deps:** update dependency gradle to v7.3.1 ([0fd0ada](https://github.com/DanySK/Alchemist-experiments-startup/commit/0fd0ada3dec2c76b4380448ea08b63488bf1c2ff))
* **deps:** update dependency gradle to v7.3.2 ([0858784](https://github.com/DanySK/Alchemist-experiments-startup/commit/0858784f9558fb8ad18524ed6712245bd0b10037))
* **deps:** update dependency gradle to v7.3.3 ([702a7ea](https://github.com/DanySK/Alchemist-experiments-startup/commit/702a7ea31643bb1c1e044dfb64f8e59ff6f64e1e))
* **deps:** update dependency gradle to v7.4 ([f7b2f02](https://github.com/DanySK/Alchemist-experiments-startup/commit/f7b2f02856ae0dd30b86df4920ef3bb90b866d32))
* **deps:** update dependency gradle to v7.4.1 ([77907fb](https://github.com/DanySK/Alchemist-experiments-startup/commit/77907fbc092c97e7144902c6edf70adf49f8287b))
* **deps:** update dependency gradle to v7.4.2 ([219e503](https://github.com/DanySK/Alchemist-experiments-startup/commit/219e50312aaee11b4c73f38d7bbd190e7ef6177c))
* **deps:** update dependency gradle to v7.5 ([e4f1202](https://github.com/DanySK/Alchemist-experiments-startup/commit/e4f1202216a80c263a810e22a0f815a795397826))
* **deps:** update dependency gradle to v7.5.1 ([75a2c8c](https://github.com/DanySK/Alchemist-experiments-startup/commit/75a2c8ca4ebde5e4115d3672d5a6ba502358ecda))
* **deps:** update dependency gradle to v7.6 ([cf5df22](https://github.com/DanySK/Alchemist-experiments-startup/commit/cf5df22f1501a3155e121d5e60ea97a25c791229))
* **deps:** update dependency it.unibo.alchemist:alchemist to v12.0.3 ([a2cfa21](https://github.com/DanySK/Alchemist-experiments-startup/commit/a2cfa2148b05bcebd1dc5d0fa39bada6a69dcc8b))
* **deps:** update dependency it.unibo.alchemist:alchemist to v13.0.3 ([7633167](https://github.com/DanySK/Alchemist-experiments-startup/commit/7633167bb89c1e9ebda2c6287c5774a261f39a40))
* **deps:** update dependency it.unibo.alchemist:alchemist to v13.0.6 ([f2f19e4](https://github.com/DanySK/Alchemist-experiments-startup/commit/f2f19e4cebbefabfcfb4440baf1adc47c7f65ad1))
* **deps:** update dependency it.unibo.alchemist:alchemist-incarnation-protelis to v12.0.4 ([fbd0a74](https://github.com/DanySK/Alchemist-experiments-startup/commit/fbd0a74a566dd47c981baf5df7bdb43b10f9a487))
* **deps:** update dependency it.unibo.alchemist:alchemist-incarnation-scafi to v12.1.7 ([f6e9704](https://github.com/DanySK/Alchemist-experiments-startup/commit/f6e9704bc27ef3b6de2a946cc031b35bdff4cf24))
* **deps:** update dependency it.unibo.alchemist:alchemist-incarnation-scafi to v13.0.4 ([7a36e21](https://github.com/DanySK/Alchemist-experiments-startup/commit/7a36e21c2968853b418d7cfa9132f897b99ba54a))
* **deps:** update dependency matplotlib to v3.5.0 ([6fedce4](https://github.com/DanySK/Alchemist-experiments-startup/commit/6fedce4e0a9b8f5d46897d53d4b4c92e3e3164af))
* **deps:** update dependency matplotlib to v3.5.1 ([005ec60](https://github.com/DanySK/Alchemist-experiments-startup/commit/005ec60a73d66ccb9f5521562294011e2ee811f6))
* **deps:** update dependency matplotlib to v3.5.2 ([b8fa622](https://github.com/DanySK/Alchemist-experiments-startup/commit/b8fa62294ca02f4cac35b9802018924715cad146))
* **deps:** update dependency matplotlib to v3.5.3 ([2937750](https://github.com/DanySK/Alchemist-experiments-startup/commit/293775015498cf425084d6019a967ae22712c1c5))
* **deps:** update dependency matplotlib to v3.6.0 ([8b261a2](https://github.com/DanySK/Alchemist-experiments-startup/commit/8b261a253460df5e0106fb8a84ca49f264781504))
* **deps:** update dependency matplotlib to v3.6.1 ([bb6219d](https://github.com/DanySK/Alchemist-experiments-startup/commit/bb6219d80b1303812da26d84cd7396728a6c3aae))
* **deps:** update dependency matplotlib to v3.6.2 ([2c3cb99](https://github.com/DanySK/Alchemist-experiments-startup/commit/2c3cb994f870559041935997030ece2cee4a3afa))
* **deps:** update dependency matplotlib to v3.6.3 ([8c446fd](https://github.com/DanySK/Alchemist-experiments-startup/commit/8c446fd048cd6a1a4258b78d270be14242e704dd))
* **deps:** update dependency openjdk to v18 ([43e2783](https://github.com/DanySK/Alchemist-experiments-startup/commit/43e278310b5797e8c6ee4fdeb54e1978e61524b4))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.7.0 ([79f1b47](https://github.com/DanySK/Alchemist-experiments-startup/commit/79f1b47ba31cda8db54ab11b0b0ef5c416d36c3e))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.7.10 ([e9e4155](https://github.com/DanySK/Alchemist-experiments-startup/commit/e9e4155eb1fcad2557a746381d9015a451ff9530))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.7.20 ([5437b3e](https://github.com/DanySK/Alchemist-experiments-startup/commit/5437b3e30d0ceb34ff9b437aa0a2d64d671d68ea))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.7.21 ([7c64e39](https://github.com/DanySK/Alchemist-experiments-startup/commit/7c64e395255d21a5c5902263f0a34beefe957404))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.7.22 ([bbeeb01](https://github.com/DanySK/Alchemist-experiments-startup/commit/bbeeb0138c169147f8818d76fb9c41e05254ff33))
* **deps:** update dependency org.jetbrains.kotlin.jvm to v1.8.0 ([95a34dd](https://github.com/DanySK/Alchemist-experiments-startup/commit/95a34dd8180aa2a689cc4256a9f2628014370ba2))
* **deps:** update dependency python to v3.10.4 ([f4d6713](https://github.com/DanySK/Alchemist-experiments-startup/commit/f4d67134b9d1f38869d8d4d6430217188739c223))
* **deps:** update dependency xarray to v0.20.2 ([17f722d](https://github.com/DanySK/Alchemist-experiments-startup/commit/17f722df695d8912f6432e0767b7fe73a790890b))
* **deps:** update dependency xarray to v0.21.0 ([255c118](https://github.com/DanySK/Alchemist-experiments-startup/commit/255c118b8d1cec68c36899ea5d7b81d585384274))
* **deps:** update dependency xarray to v0.21.1 ([62f342d](https://github.com/DanySK/Alchemist-experiments-startup/commit/62f342d5ad872c2783f9ca1357a925ec668acf2a))
* **deps:** update dependency xarray to v2022 ([29e742b](https://github.com/DanySK/Alchemist-experiments-startup/commit/29e742b0a246048c584a130bab2b81ece70c448a))
* **deps:** update dependency xarray to v2022.10.0 ([bc3ef08](https://github.com/DanySK/Alchemist-experiments-startup/commit/bc3ef08bf8895dffd04f4b1ae8eda8f9546fdf13))
* **deps:** update dependency xarray to v2022.11.0 ([45c4b38](https://github.com/DanySK/Alchemist-experiments-startup/commit/45c4b389d46f5f0904aacef519972fa7cf94cd9e))
* **deps:** update dependency xarray to v2022.12.0 ([e032eab](https://github.com/DanySK/Alchemist-experiments-startup/commit/e032eabae1ea7d6b516ee9fedd34c3b4372cb825))
* **deps:** update dependency xarray to v2022.6.0 ([7d278b6](https://github.com/DanySK/Alchemist-experiments-startup/commit/7d278b6a3cff33f1262797ebc1df03b8eedf0386))
* **deps:** update dependency xarray to v2022.9.0 ([f81445d](https://github.com/DanySK/Alchemist-experiments-startup/commit/f81445d91b3136c9c68bdb8981e6d17b2b824907))
* **deps:** update it.unibo.alchemist ([ec982b1](https://github.com/DanySK/Alchemist-experiments-startup/commit/ec982b13114dbfde7ac8f50b99b0d9c42637dff3))
* **deps:** update it.unibo.alchemist to v12 ([b23827c](https://github.com/DanySK/Alchemist-experiments-startup/commit/b23827c2e3352c31e05e837477112f57939b022a))
* **deps:** update it.unibo.alchemist to v12.1.2 ([d3552b0](https://github.com/DanySK/Alchemist-experiments-startup/commit/d3552b02a26fcf3a60675ff1280ee553e6450b15))
* **deps:** update it.unibo.alchemist to v12.1.3 ([6adf430](https://github.com/DanySK/Alchemist-experiments-startup/commit/6adf43017abb5df131e35ff6ceb029ec65f74951))
* **deps:** update it.unibo.alchemist to v12.1.4 ([f9b8a53](https://github.com/DanySK/Alchemist-experiments-startup/commit/f9b8a538458c374bd858d324f7ca264f40ccfd3b))
* **deps:** update it.unibo.alchemist to v12.1.5 ([5e1b2ab](https://github.com/DanySK/Alchemist-experiments-startup/commit/5e1b2ab5a1c3b3abccaf33674097302ec60a0e0f))
* **deps:** update it.unibo.alchemist to v12.1.6 ([5bc3492](https://github.com/DanySK/Alchemist-experiments-startup/commit/5bc34926e79881e0de8b21df6fde415d27d438bb))
* **deps:** update it.unibo.alchemist to v12.2.0 ([841b7f5](https://github.com/DanySK/Alchemist-experiments-startup/commit/841b7f594061c3fc177c25c2e6743080efb599c8))
* **deps:** update it.unibo.alchemist to v12.2.1 ([58a2e7f](https://github.com/DanySK/Alchemist-experiments-startup/commit/58a2e7f3640fa5b566041cd79bd39b7d01e17cde))
* **deps:** update it.unibo.alchemist to v12.2.10 ([81d9e84](https://github.com/DanySK/Alchemist-experiments-startup/commit/81d9e84f84e4a8e3f334ad499346aaa02cc2f159))
* **deps:** update it.unibo.alchemist to v12.2.11 ([2c8228d](https://github.com/DanySK/Alchemist-experiments-startup/commit/2c8228d25518357ed27648ee71c59c979900223f))
* **deps:** update it.unibo.alchemist to v12.2.7 ([9b6fedc](https://github.com/DanySK/Alchemist-experiments-startup/commit/9b6fedc7e498d0c691d25ee31df57281f72eaeb2))
* **deps:** update it.unibo.alchemist to v12.2.8 ([c2011eb](https://github.com/DanySK/Alchemist-experiments-startup/commit/c2011eb89c391480ec5f9c54a7c6c24a5347e425))
* **deps:** update it.unibo.alchemist to v12.2.9 ([523b91e](https://github.com/DanySK/Alchemist-experiments-startup/commit/523b91ebe35055987a5e5d5546988a43c8de9d55))
* **deps:** update it.unibo.alchemist to v13 ([d378116](https://github.com/DanySK/Alchemist-experiments-startup/commit/d378116296ef7fcb4996fb5e58bfb96326333b73))
* **deps:** update it.unibo.alchemist to v13.0.1 ([9140444](https://github.com/DanySK/Alchemist-experiments-startup/commit/9140444a7be9f0fe6afb8e75b6a03ee0969e211d))
* **deps:** update it.unibo.alchemist to v13.0.2 ([53d19c6](https://github.com/DanySK/Alchemist-experiments-startup/commit/53d19c64ccbc16cf48a8518e1991b647dae4bc7b))
* **deps:** update it.unibo.alchemist to v13.0.5 ([612f656](https://github.com/DanySK/Alchemist-experiments-startup/commit/612f656d0a2f3c30b8e6b5ab414fda28df03995c))
* **deps:** update it.unibo.alchemist to v14 ([b0e5e2d](https://github.com/DanySK/Alchemist-experiments-startup/commit/b0e5e2d69687337203a045d6085203e98ba80cf3))
* **deps:** update it.unibo.alchemist to v14.0.1 ([7ddb5bc](https://github.com/DanySK/Alchemist-experiments-startup/commit/7ddb5bc2f9da615572bcbc8f5e98d59b482241b3))
* **deps:** update it.unibo.alchemist to v14.0.2 ([2514a57](https://github.com/DanySK/Alchemist-experiments-startup/commit/2514a5717385b7a125808053c270568a362496d0))
* **deps:** update it.unibo.alchemist to v14.1.0 ([26f2cf2](https://github.com/DanySK/Alchemist-experiments-startup/commit/26f2cf2f8870cd06deb059d6d361bf2b19dd1ef8))
* **deps:** update it.unibo.alchemist to v15 ([19668e9](https://github.com/DanySK/Alchemist-experiments-startup/commit/19668e96caab03bff6028033157fda5912454b30))
* **deps:** update it.unibo.alchemist to v15.0.2 ([fee8256](https://github.com/DanySK/Alchemist-experiments-startup/commit/fee825694d769f5a6abe611309f3d0eb9643c276))
* **deps:** update it.unibo.alchemist to v15.0.3 ([17fe54c](https://github.com/DanySK/Alchemist-experiments-startup/commit/17fe54cf33fa96e25060ba084ffda3c93e636793))
* **deps:** update it.unibo.alchemist to v15.0.4 ([7ac83ed](https://github.com/DanySK/Alchemist-experiments-startup/commit/7ac83ed6db6ce22537f78258a6a2f9cc9508c6d9))
* **deps:** update it.unibo.alchemist to v16 ([e89b90e](https://github.com/DanySK/Alchemist-experiments-startup/commit/e89b90e664d809bace7a295b1f4691ea0fdba630))
* **deps:** update it.unibo.alchemist to v17 ([2174d33](https://github.com/DanySK/Alchemist-experiments-startup/commit/2174d3313a78092d368864631b7edd779d02a0b3))
* **deps:** update it.unibo.alchemist to v17.0.1 ([92b76f0](https://github.com/DanySK/Alchemist-experiments-startup/commit/92b76f07582c7db224ab4674113c673a9f06f5d5))
* **deps:** update it.unibo.alchemist to v17.0.2 ([e9ed64d](https://github.com/DanySK/Alchemist-experiments-startup/commit/e9ed64d1e6b01d438db1d4c8d7754c04d7e1ca94))
* **deps:** update it.unibo.alchemist to v17.0.4 ([00b0890](https://github.com/DanySK/Alchemist-experiments-startup/commit/00b0890e14d1ba4fd43f9ecd0d4a05e32db9bde4))
* **deps:** update plugin com.gradle.enterprise to v3.10 ([ec84f24](https://github.com/DanySK/Alchemist-experiments-startup/commit/ec84f24b85438a672a8c15ed36e1fa06410ff14c))
* **deps:** update plugin com.gradle.enterprise to v3.10.1 ([dc0eb15](https://github.com/DanySK/Alchemist-experiments-startup/commit/dc0eb156b0f9e7eacd898627b17ef22442efb539))
* **deps:** update plugin com.gradle.enterprise to v3.10.2 ([540fb9e](https://github.com/DanySK/Alchemist-experiments-startup/commit/540fb9e22d52d7edf0228e1730b2d4c0d37836df))
* **deps:** update plugin com.gradle.enterprise to v3.10.3 ([93639c8](https://github.com/DanySK/Alchemist-experiments-startup/commit/93639c854cd8917efe5fca7ae8320ec064c062f1))
* **deps:** update plugin com.gradle.enterprise to v3.11 ([39303c0](https://github.com/DanySK/Alchemist-experiments-startup/commit/39303c0da12622934bc8e47ea56acc616cc764da))
* **deps:** update plugin com.gradle.enterprise to v3.11.1 ([377af6a](https://github.com/DanySK/Alchemist-experiments-startup/commit/377af6a419a3a5638b59a11824c8304364b567de))
* **deps:** update plugin com.gradle.enterprise to v3.7.2 ([9d81f38](https://github.com/DanySK/Alchemist-experiments-startup/commit/9d81f388168eaa5b1d837bccc41390d9c35af922))
* **deps:** update plugin com.gradle.enterprise to v3.8 ([c8e51a1](https://github.com/DanySK/Alchemist-experiments-startup/commit/c8e51a17c19f398df874e5bf56b63d45af0cedc0))
* **deps:** update plugin com.gradle.enterprise to v3.8.1 ([36c0696](https://github.com/DanySK/Alchemist-experiments-startup/commit/36c0696ba4bf633bb3b4eef8531795f7566c42fe))
* **deps:** update plugin com.gradle.enterprise to v3.9 ([d5709bd](https://github.com/DanySK/Alchemist-experiments-startup/commit/d5709bdd5bd23eac4ca586930d951a21f03e1f06))
* **deps:** update plugin kotlin-jvm to v1.6.0 ([f8e974e](https://github.com/DanySK/Alchemist-experiments-startup/commit/f8e974e861aef31157c3581c7fc8b59d55da3f36))
* **deps:** update plugin kotlin-jvm to v1.6.10 ([bcaf3d2](https://github.com/DanySK/Alchemist-experiments-startup/commit/bcaf3d2487aaeeb2df9e5fcd3b3f259218d2f27d))
* **deps:** update plugin kotlin-jvm to v1.6.20 ([62c8e30](https://github.com/DanySK/Alchemist-experiments-startup/commit/62c8e309d221950b6c550a5c8377e5c5db0d3bfc))
* **deps:** update plugin kotlin-jvm to v1.6.21 ([db462dc](https://github.com/DanySK/Alchemist-experiments-startup/commit/db462dcf7610cdec80014d73b19d8e6610847768))
* **deps:** update plugin kotlin-qa to v0.10.0 ([de04a9e](https://github.com/DanySK/Alchemist-experiments-startup/commit/de04a9e3e83b07eb6a6ffcc4b24abda90233f2b7))
* **deps:** update plugin kotlin-qa to v0.10.1 ([672035b](https://github.com/DanySK/Alchemist-experiments-startup/commit/672035b121e00ded00dcb72e8b6558d860269d89))
* **deps:** update plugin kotlin-qa to v0.12.0 ([555a314](https://github.com/DanySK/Alchemist-experiments-startup/commit/555a3144327f7b0171a349489afd72a7b3ca2d58))
* **deps:** update plugin kotlin-qa to v0.12.1 ([3a24bfe](https://github.com/DanySK/Alchemist-experiments-startup/commit/3a24bfe93330f90441f4fa72f5ea683001a717c3))
* **deps:** update plugin kotlin-qa to v0.13.0 ([1b700a7](https://github.com/DanySK/Alchemist-experiments-startup/commit/1b700a78ea6495bf6f3a692a84ee0685de02853f))
* **deps:** update plugin kotlin-qa to v0.14.0 ([a1102a3](https://github.com/DanySK/Alchemist-experiments-startup/commit/a1102a32e1f5d8b1b95fc2dedbea34e4408c1633))
* **deps:** update plugin kotlin-qa to v0.14.1 ([ebd110e](https://github.com/DanySK/Alchemist-experiments-startup/commit/ebd110e2b8cf59e7d447e8d0fba08ddf83a0a23c))
* **deps:** update plugin kotlin-qa to v0.14.2 ([b7a65a7](https://github.com/DanySK/Alchemist-experiments-startup/commit/b7a65a781351ded97da5368860e6511a8c990dff))
* **deps:** update plugin kotlin-qa to v0.15.0 ([cfe3149](https://github.com/DanySK/Alchemist-experiments-startup/commit/cfe31497e4371435638cdd09c4149e6480efacf7))
* **deps:** update plugin kotlin-qa to v0.15.1 ([ede2be3](https://github.com/DanySK/Alchemist-experiments-startup/commit/ede2be36f8f56f5928dd50175467ef2363592760))
* **deps:** update plugin kotlin-qa to v0.16.0 ([588dd5f](https://github.com/DanySK/Alchemist-experiments-startup/commit/588dd5fc0247989a0fa677eece61751ca72f9890))
* **deps:** update plugin kotlin-qa to v0.16.1 ([91ba758](https://github.com/DanySK/Alchemist-experiments-startup/commit/91ba7587e6b1895092e28258a660222e79d1c185))
* **deps:** update plugin kotlin-qa to v0.18.0 ([2db181e](https://github.com/DanySK/Alchemist-experiments-startup/commit/2db181ef56fa17d9016f0f8e7d6c07adebdca705))
* **deps:** update plugin kotlin-qa to v0.19.0 ([404a320](https://github.com/DanySK/Alchemist-experiments-startup/commit/404a32037ce4155419ea20cbc3e7749a35245508))
* **deps:** update plugin kotlin-qa to v0.19.1 ([accc86a](https://github.com/DanySK/Alchemist-experiments-startup/commit/accc86ae1aef39c8136db17aaa7609796c956c3a))
* **deps:** update plugin kotlin-qa to v0.20.3 ([20b6c3a](https://github.com/DanySK/Alchemist-experiments-startup/commit/20b6c3a70ba545d93e8ca1c20f23dec9d27e0060))
* **deps:** update plugin kotlin-qa to v0.20.4 ([7c80ac8](https://github.com/DanySK/Alchemist-experiments-startup/commit/7c80ac8a889a8eda61cb94dcd57875ed01e574a0))
* **deps:** update plugin kotlin-qa to v0.21.0 ([32ee127](https://github.com/DanySK/Alchemist-experiments-startup/commit/32ee127ffaae2c1795d7ebc83c3ad468529fad92))
* **deps:** update plugin kotlin-qa to v0.22.0 ([58b0895](https://github.com/DanySK/Alchemist-experiments-startup/commit/58b0895e6dd89f2c5be46b6f048bf43fc4a2c65d))
* **deps:** update plugin kotlin-qa to v0.22.1 ([80b81ae](https://github.com/DanySK/Alchemist-experiments-startup/commit/80b81ae42952d9b3fa9468ff02049c027c41e5ff))
* **deps:** update plugin kotlin-qa to v0.22.2 ([4eb7ca9](https://github.com/DanySK/Alchemist-experiments-startup/commit/4eb7ca953c1029cf8bbda6bc339307c9eea5ebe7))
* **deps:** update plugin kotlin-qa to v0.23.0 ([d8ad7ad](https://github.com/DanySK/Alchemist-experiments-startup/commit/d8ad7ad8b6b7734ceaafd5983c04de2ad0d1386d))
* **deps:** update plugin kotlin-qa to v0.23.1 ([a8d508a](https://github.com/DanySK/Alchemist-experiments-startup/commit/a8d508aff0cd04ec94d80f75b5916fc50fbe03bd))
* **deps:** update plugin kotlin-qa to v0.23.2 ([02d79e9](https://github.com/DanySK/Alchemist-experiments-startup/commit/02d79e986fb15aef33df920e6b517d6958f0da78))
* **deps:** update plugin kotlin-qa to v0.24.0 ([4e83b40](https://github.com/DanySK/Alchemist-experiments-startup/commit/4e83b40b12f49c366fc18a9301305498cf6caded))
* **deps:** update plugin kotlin-qa to v0.25.0 ([790e4a4](https://github.com/DanySK/Alchemist-experiments-startup/commit/790e4a4e32453fdb27aad7340d3b24b7028c5d5a))
* **deps:** update plugin kotlin-qa to v0.25.1 ([6fcd004](https://github.com/DanySK/Alchemist-experiments-startup/commit/6fcd004a27e8fa62cbd675694420987906c3b646))
* **deps:** update plugin kotlin-qa to v0.26.0 ([3f7147d](https://github.com/DanySK/Alchemist-experiments-startup/commit/3f7147dcc3845fc797bd1cce1d7fcccc59ed059c))
* **deps:** update plugin kotlin-qa to v0.26.1 ([6f3fb07](https://github.com/DanySK/Alchemist-experiments-startup/commit/6f3fb075a037bb201c85386c9aba8604207585fb))
* **deps:** update plugin kotlin-qa to v0.27.0 ([07ce777](https://github.com/DanySK/Alchemist-experiments-startup/commit/07ce77747b4c1b73a19afafc2130481ba0825976))
* **deps:** update plugin kotlin-qa to v0.27.1 ([4eba7e3](https://github.com/DanySK/Alchemist-experiments-startup/commit/4eba7e3277d0d93b96f9bb748a862ade29250781))
* **deps:** update plugin kotlin-qa to v0.28.0 ([52b6e27](https://github.com/DanySK/Alchemist-experiments-startup/commit/52b6e274164ac5f9723a146db4771667bcba2510))
* **deps:** update plugin kotlin-qa to v0.29.0 ([51f7ca3](https://github.com/DanySK/Alchemist-experiments-startup/commit/51f7ca385bc53abda5eb436d2a06a67635d97328))
* **deps:** update plugin kotlin-qa to v0.29.1 ([0f8edc6](https://github.com/DanySK/Alchemist-experiments-startup/commit/0f8edc62ab6664368f6992f6f0547ec3cdcc59cf))
* **deps:** update plugin kotlin-qa to v0.29.2 ([05bbfd7](https://github.com/DanySK/Alchemist-experiments-startup/commit/05bbfd7e4399dc12d3f76157927370e50ca8c93b))
* **deps:** update plugin kotlin-qa to v0.31.0 ([8cf8127](https://github.com/DanySK/Alchemist-experiments-startup/commit/8cf8127ec08544f7832921583f0f92103c694178))
* **deps:** update plugin kotlin-qa to v0.6.0 ([6be1e1e](https://github.com/DanySK/Alchemist-experiments-startup/commit/6be1e1e14e2616e31f752bed2c2bf69cfbc893b3))
* **deps:** update plugin kotlin-qa to v0.7.0 ([5ca5d1d](https://github.com/DanySK/Alchemist-experiments-startup/commit/5ca5d1d91c3e42631b2af5b0be33d9004ea03f8f))
* **deps:** update plugin kotlin-qa to v0.7.1 ([9319000](https://github.com/DanySK/Alchemist-experiments-startup/commit/9319000ac69b75c5e259216b7b5f57096f679512))
* **deps:** update plugin kotlin-qa to v0.8.0 ([b9c4870](https://github.com/DanySK/Alchemist-experiments-startup/commit/b9c4870dbc440b10c25bc0020fc979559c85ebb0))
* **deps:** update plugin kotlin-qa to v0.8.1 ([b81c36f](https://github.com/DanySK/Alchemist-experiments-startup/commit/b81c36f0f790f377babf8047dfab29d3fbd000df))
* **deps:** update plugin kotlin-qa to v0.8.2 ([e5f44f3](https://github.com/DanySK/Alchemist-experiments-startup/commit/e5f44f314560d0f8c9e3b674fc562260f764ca69))
* **deps:** update plugin kotlin-qa to v0.8.3 ([894f1f9](https://github.com/DanySK/Alchemist-experiments-startup/commit/894f1f97766f28c3913e8564070ea78488767e77))
* **deps:** update plugin kotlin-qa to v0.9.0 ([684ca18](https://github.com/DanySK/Alchemist-experiments-startup/commit/684ca18945fd2071c1e1bb1ef021e64d7f5c7a7e))
* **deps:** update plugin multijvmtesting to v0.2.0 ([42dee20](https://github.com/DanySK/Alchemist-experiments-startup/commit/42dee200a3100e1b7876daec1ed9cce6a910dbff))
* **deps:** update plugin multijvmtesting to v0.2.1 ([45964c5](https://github.com/DanySK/Alchemist-experiments-startup/commit/45964c568352c7f7425cc5a4e6ffc9f2513eda4e))
* **deps:** update plugin multijvmtesting to v0.2.2 ([e13d083](https://github.com/DanySK/Alchemist-experiments-startup/commit/e13d0838a202bf2732510b856b753983fcd6c441))
* **deps:** update plugin multijvmtesting to v0.3.0 ([74c8c3b](https://github.com/DanySK/Alchemist-experiments-startup/commit/74c8c3b7ae67e6ad33138d99ed7c194a72680334))
* **deps:** update plugin multijvmtesting to v0.3.1 ([ba70006](https://github.com/DanySK/Alchemist-experiments-startup/commit/ba70006cc17ae665725f72fe83b87a97ba1fadb3))
* **deps:** update plugin multijvmtesting to v0.3.2 ([d34d6ca](https://github.com/DanySK/Alchemist-experiments-startup/commit/d34d6ca93e56b25b729765a7b4195fa5b15e1e5a))
* **deps:** update plugin multijvmtesting to v0.3.3 ([847eee9](https://github.com/DanySK/Alchemist-experiments-startup/commit/847eee948f9c2dcfbaa224242ddc6fecbe16d7e0))
* **deps:** update plugin multijvmtesting to v0.3.4 ([6696eb9](https://github.com/DanySK/Alchemist-experiments-startup/commit/6696eb936cee87d3930cbd0dc1968156ee115148))
* **deps:** update plugin multijvmtesting to v0.3.5 ([9231d50](https://github.com/DanySK/Alchemist-experiments-startup/commit/9231d508b35b999c1bbee84b201fbb5f3a550c87))
* **deps:** update plugin multijvmtesting to v0.3.6 ([6f148e1](https://github.com/DanySK/Alchemist-experiments-startup/commit/6f148e10ffc0aa662d7bc089ea6cbd6f262ea07b))
* **deps:** update plugin multijvmtesting to v0.3.7 ([edfe125](https://github.com/DanySK/Alchemist-experiments-startup/commit/edfe1251670a8e5ae97fb21f0b6754c44fb7f2fe))
* **deps:** update plugin multijvmtesting to v0.4.0 ([019b8d6](https://github.com/DanySK/Alchemist-experiments-startup/commit/019b8d601893ff635a670e02c72db458ed4bf79e))
* **deps:** update plugin multijvmtesting to v0.4.10 ([5532ae5](https://github.com/DanySK/Alchemist-experiments-startup/commit/5532ae5ed538f7b109624f52a33bb91b4d59c2bb))
* **deps:** update plugin multijvmtesting to v0.4.11 ([5f9b2bd](https://github.com/DanySK/Alchemist-experiments-startup/commit/5f9b2bdb8b4e09a2b3452e4945fe4bc01feecfdb))
* **deps:** update plugin multijvmtesting to v0.4.12 ([2d85f9a](https://github.com/DanySK/Alchemist-experiments-startup/commit/2d85f9ad7033cb8304b30dca8f63bd3c5404ba2a))
* **deps:** update plugin multijvmtesting to v0.4.13 ([c02123a](https://github.com/DanySK/Alchemist-experiments-startup/commit/c02123ab6266c91491edf5f940096a2e01f7f750))
* **deps:** update plugin multijvmtesting to v0.4.14 ([2528431](https://github.com/DanySK/Alchemist-experiments-startup/commit/252843130a6685532fa6d2b9473d2c417f8d7a98))
* **deps:** update plugin multijvmtesting to v0.4.15 ([96fb9b5](https://github.com/DanySK/Alchemist-experiments-startup/commit/96fb9b53f689e129968c6fa79cfb189202d444ed))
* **deps:** update plugin multijvmtesting to v0.4.2 ([ba3a73a](https://github.com/DanySK/Alchemist-experiments-startup/commit/ba3a73a90b1ccdd86504c1e04c895c72d5c0ba3e))
* **deps:** update plugin multijvmtesting to v0.4.3 ([ef7d9b8](https://github.com/DanySK/Alchemist-experiments-startup/commit/ef7d9b85cb5cb10c50552f3c08249a2d8f887b47))
* **deps:** update plugin multijvmtesting to v0.4.4 ([9a84920](https://github.com/DanySK/Alchemist-experiments-startup/commit/9a849209036728d7e590cd3828f6d77bcc185879))
* **deps:** update plugin multijvmtesting to v0.4.5 ([3e8720b](https://github.com/DanySK/Alchemist-experiments-startup/commit/3e8720b6654bf692ce347d49c052a1448dd4c89a))
* **deps:** update plugin multijvmtesting to v0.4.6 ([168f727](https://github.com/DanySK/Alchemist-experiments-startup/commit/168f7272229926cb00cf369bea890135ad040f4b))
* **deps:** update plugin multijvmtesting to v0.4.7 ([0935925](https://github.com/DanySK/Alchemist-experiments-startup/commit/0935925bf1bb9577823753e87d6e4785641d9d6e))
* **deps:** update plugin multijvmtesting to v0.4.8 ([62c0918](https://github.com/DanySK/Alchemist-experiments-startup/commit/62c0918a4790a214eba1491acfa5b602777e0a12))
* **deps:** update plugin multijvmtesting to v0.4.9 ([14648ea](https://github.com/DanySK/Alchemist-experiments-startup/commit/14648eaf59eac187d7f8e91e850cd5a848e9a541))


### Build and continuous integration

* add missing checkout ([717cb24](https://github.com/DanySK/Alchemist-experiments-startup/commit/717cb240fc603f78e162bf8da310f47ba6714658))
* add pipeline summary job ([d7ed0ed](https://github.com/DanySK/Alchemist-experiments-startup/commit/d7ed0ed194ed307d6032a25136dcd67dd1d8088a))
* add the mergify config from Alchemist ([dc1905a](https://github.com/DanySK/Alchemist-experiments-startup/commit/dc1905a5ff8eb3d8f32bbb91699ef038312d9ff5))
* allow release updates in case of re-runs ([eb85b63](https://github.com/DanySK/Alchemist-experiments-startup/commit/eb85b633407d9cd28deb268ff28c405ae451e3b0))
* create .python-version ([34a9263](https://github.com/DanySK/Alchemist-experiments-startup/commit/34a926301656bcc9543e1353f9d4772608dacec0))
* **deps:** update actions/cache action to v3 ([9413833](https://github.com/DanySK/Alchemist-experiments-startup/commit/94138335d5a21ba48eeae9feac2d1f9e4372f3d9))
* **deps:** update actions/checkout action to v3.0.2 ([249cb0b](https://github.com/DanySK/Alchemist-experiments-startup/commit/249cb0bfea74e44d857f169ab6dac00c64534b62))
* **deps:** update actions/checkout action to v3.1.0 ([c7de7d2](https://github.com/DanySK/Alchemist-experiments-startup/commit/c7de7d2c691d9c3378407a3c2ceb556cd8b2dc45))
* **deps:** update actions/checkout action to v3.2.0 ([80f5c95](https://github.com/DanySK/Alchemist-experiments-startup/commit/80f5c952f28806839dd85a840cee187ff11fa1fd))
* **deps:** update actions/checkout action to v3.3.0 ([a2711e6](https://github.com/DanySK/Alchemist-experiments-startup/commit/a2711e6aa739a60aab291bb3d34eaa7e7237ae7b))
* **deps:** update actions/setup-python action to v3 ([a1bafd2](https://github.com/DanySK/Alchemist-experiments-startup/commit/a1bafd298800ff2977c3994ce1a39e10d38f3edc))
* **deps:** update danysk/action-checkout action to v0.2.1 ([9851f1a](https://github.com/DanySK/Alchemist-experiments-startup/commit/9851f1a9cef812c2c55d912668b5b1b1b23bc82d))
* **deps:** update danysk/action-checkout action to v0.2.2 ([212e6b3](https://github.com/DanySK/Alchemist-experiments-startup/commit/212e6b3889991b2b0dc556991c7fd6757f3d7e42))
* **deps:** update danysk/action-checkout action to v0.2.3 ([b1435e3](https://github.com/DanySK/Alchemist-experiments-startup/commit/b1435e32fb297f042a054f2e7989ee9fc878958f))
* **deps:** update danysk/action-checkout action to v0.2.4 ([c072c7d](https://github.com/DanySK/Alchemist-experiments-startup/commit/c072c7d5fb3fdceb2b47173d2b125ab285cb6601))
* **deps:** update danysk/action-checkout action to v0.2.5 ([66265ec](https://github.com/DanySK/Alchemist-experiments-startup/commit/66265ece7c2341a63941ca7f84bc9c38ac10bf56))
* **deps:** update danysk/build-check-deploy-gradle-action action to v1.2.15 ([4b12b50](https://github.com/DanySK/Alchemist-experiments-startup/commit/4b12b50ea93ad5444a3da8a1ebf34a683567a5c3))
* **deps:** update danysk/build-check-deploy-gradle-action action to v1.2.16 ([8c8ae77](https://github.com/DanySK/Alchemist-experiments-startup/commit/8c8ae773659b080bcb106b2647b8530d3dab3314))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2 ([c8db7aa](https://github.com/DanySK/Alchemist-experiments-startup/commit/c8db7aa052f5825a9d9c59de258d5ecd6a3c8f8a))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.0.1 ([d34c473](https://github.com/DanySK/Alchemist-experiments-startup/commit/d34c473463ef2664e29a7aa26032c12925e136f6))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.0.2 ([8c89e08](https://github.com/DanySK/Alchemist-experiments-startup/commit/8c89e08c1cab3daad604b678dbf7dd0c9da73a75))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.0.3 ([003595d](https://github.com/DanySK/Alchemist-experiments-startup/commit/003595d73f8dd0df24b0e36b1db93f738dc77c7c))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.0 ([4d83e97](https://github.com/DanySK/Alchemist-experiments-startup/commit/4d83e97f4da4670d3c3bcb905f882e806c1d814b))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.1 ([ebb39c1](https://github.com/DanySK/Alchemist-experiments-startup/commit/ebb39c1a23c56791de8ab16e3dac378497d659cd))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.10 ([f8eeb58](https://github.com/DanySK/Alchemist-experiments-startup/commit/f8eeb58179dda2cc76da3f8009fbe0a1e18a4a2e))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.11 ([1891808](https://github.com/DanySK/Alchemist-experiments-startup/commit/1891808d79c80965e79261692a0c702b34bfb970))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.12 ([60abb21](https://github.com/DanySK/Alchemist-experiments-startup/commit/60abb21d188e57f5a06cfc97ce0372f8443c2b48))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.13 ([26c174a](https://github.com/DanySK/Alchemist-experiments-startup/commit/26c174a3d7eb64f458eb8cbcf4e2ad1985e5b574))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.14 ([bb7ec31](https://github.com/DanySK/Alchemist-experiments-startup/commit/bb7ec31c6331611c697a5ac7e4d8129f65d67a5e))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.15 ([cd750a1](https://github.com/DanySK/Alchemist-experiments-startup/commit/cd750a116f08000dca16cbc3be56a03ade987a58))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.16 ([ce873e6](https://github.com/DanySK/Alchemist-experiments-startup/commit/ce873e61160ab150d7a91b24d66b65aafd9bd319))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.17 ([06227b0](https://github.com/DanySK/Alchemist-experiments-startup/commit/06227b0356812ca8cc10cf11aec48674e3975fa4))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.18 ([1c62c01](https://github.com/DanySK/Alchemist-experiments-startup/commit/1c62c01eee12cb3e9cbae41fa89c58fdf7fff98d))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.19 ([4bde933](https://github.com/DanySK/Alchemist-experiments-startup/commit/4bde933dd11e6af7fc28c71dade92db33e4abf5a))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.2 ([732c44f](https://github.com/DanySK/Alchemist-experiments-startup/commit/732c44f9d3779168a7df2884e5418ce144b84751))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.20 ([4556faa](https://github.com/DanySK/Alchemist-experiments-startup/commit/4556faab5b5e49f412de9c801fab3caa65145508))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.21 ([a201334](https://github.com/DanySK/Alchemist-experiments-startup/commit/a2013343d7942371add70856fc92d0a34c489ea1))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.3 ([ce5b522](https://github.com/DanySK/Alchemist-experiments-startup/commit/ce5b522750372a56c6e4f982582415c89c1d2afb))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.4 ([b2080f1](https://github.com/DanySK/Alchemist-experiments-startup/commit/b2080f176adec0d4d7e79ff045101ef15e283f33))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.5 ([3d31154](https://github.com/DanySK/Alchemist-experiments-startup/commit/3d311544a477889589ebe5330688189ca8069036))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.6 ([93d0390](https://github.com/DanySK/Alchemist-experiments-startup/commit/93d0390a85af99f841a917e808b3cc58bd3e7e7b))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.7 ([9a4496f](https://github.com/DanySK/Alchemist-experiments-startup/commit/9a4496ff104fd5d5e8b00e591726c5f17a047975))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.8 ([f14fe7d](https://github.com/DanySK/Alchemist-experiments-startup/commit/f14fe7dcdc6b6d6b8436f48c59bfec41c9759287))
* **deps:** update danysk/build-check-deploy-gradle-action action to v2.1.9 ([ff2ba6e](https://github.com/DanySK/Alchemist-experiments-startup/commit/ff2ba6ee3a284b65848faf6945daa32b43ded171))
* **deps:** update gabrielfalcao/pyenv-action action to v10 ([5d87ce8](https://github.com/DanySK/Alchemist-experiments-startup/commit/5d87ce8486e0245db464a4f6069827d7134abd1f))
* **deps:** update gabrielfalcao/pyenv-action action to v11 ([97a8ae6](https://github.com/DanySK/Alchemist-experiments-startup/commit/97a8ae6669f7bc7223cc30b882a8e94ea73ef3c7))
* **deps:** update ncipollo/release-action action to v1.10.0 ([8603573](https://github.com/DanySK/Alchemist-experiments-startup/commit/8603573b92d23dc87a0c8409406b59138c3a7385))
* **deps:** update ncipollo/release-action action to v1.10.1 ([906a6af](https://github.com/DanySK/Alchemist-experiments-startup/commit/906a6af4fa8fbad9f8b23251a451e6f5304c242d))
* **deps:** update ncipollo/release-action action to v1.11.0 ([483dd66](https://github.com/DanySK/Alchemist-experiments-startup/commit/483dd668ad497ab0070501d1611a9605a77089d1))
* **deps:** update ncipollo/release-action action to v1.11.1 ([e0633ce](https://github.com/DanySK/Alchemist-experiments-startup/commit/e0633ceb47fd45d56dcc86c238277e2c60935376))
* **deps:** update ncipollo/release-action action to v1.11.2 ([620a298](https://github.com/DanySK/Alchemist-experiments-startup/commit/620a298b321058038c5c3615b8507c01aac44f4f))
* **deps:** update ncipollo/release-action action to v1.12.0 ([45bd721](https://github.com/DanySK/Alchemist-experiments-startup/commit/45bd721c65861dd991d51cb1523b1d2d47cce8fe))
* disable codecov ([a25f6cf](https://github.com/DanySK/Alchemist-experiments-startup/commit/a25f6cfb25860945d3f86cd05da1caba50b386fe))
* lock the version of ncipollo/release-action ([c575905](https://github.com/DanySK/Alchemist-experiments-startup/commit/c5759051e6016a04477b178d088de58e672ba9ec))
* re-enable the multi-OS ([d866012](https://github.com/DanySK/Alchemist-experiments-startup/commit/d866012f5df216b874b38ed100605b4e891fb1bd))
* revamp the build ([15412b0](https://github.com/DanySK/Alchemist-experiments-startup/commit/15412b0f7342c441db3bb4b9f4f9563634a75030))
* switch to actions/setup-python ([52e6d41](https://github.com/DanySK/Alchemist-experiments-startup/commit/52e6d41699e3b7a691ba15f527817d2fe1109437))
* switch to pyenv ([b65d7dc](https://github.com/DanySK/Alchemist-experiments-startup/commit/b65d7dc3a57ec5be0f274ada921da0266be9928a))
* switch to the stable action for Gradle jobs ([9ceba67](https://github.com/DanySK/Alchemist-experiments-startup/commit/9ceba673a2ba1482bd2cce020c537907f425c1a6))
