#!/usr/bin/env sh
DESTINATION="$HOME/Downloads/field-vmc-basic-example-$(date --utc "+%F-%H.%M.%S")"
git clone https://github.com/angelacorte/fieldvmc.git "$DESTINATION"
cd "$DESTINATION"
./gradlew runSelfConstructionGraphic