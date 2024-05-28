# run all scripts in processing

# 1. running processing
echo "Running processing"
python processing/phrases.py
python processing/produce_annotated_csv.py
python processing/stylistic_features.py
python processing/upos_tags.py
python processing/word_use.py

# 2. update the visualisations
echo "Updating the visualisations"
python visualization/phrases.py
python visualization/word_use.py
python visualization/style.py
python visualization/upos_tags.py

# 3. update the website
echo "Updating the website"
mkdocs build
mkdocs gh-deploy --force
