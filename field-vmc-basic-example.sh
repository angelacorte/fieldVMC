#!/usr/bin/env sh
DESTINATION="$HOME/Downloads/field-vmc-basic-example-$(date --utc "+%F-%H.%M.%S")"
git clone https://github.com/angelacorte/fieldVMC.git "$DESTINATION"
cd "$DESTINATION"
MAX_SEED=0 ./gradlew runSelfConstructionFieldVMCGraphic