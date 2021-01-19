.PHONY: sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = s3://ojd-mapping-career-causeways/
PROFILE = 'n/a'

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# to delete
# ## Fetch data and sync raw to s3
# fetch:
# 	$(PYTHON_INTERPRETER) {{ cookiecutter.repo_name }}/fetch_data.py
# 	make sync_data_to_s3
#
# ## Sync raw from s3 and make Dataset
# data: sync_data_from_s3
# 	$(PYTHON_INTERPRETER) {{ cookiecutter.repo_name }}/make_dataset.py

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/raw s3://$(BUCKET)/data/raw
	aws s3 sync data/interim s3://$(BUCKET)/data/interim
	aws s3 sync data/processed s3://$(BUCKET)/data/processed
else
	aws s3 sync data/raw s3://$(BUCKET)/data/raw --profile $(PROFILE)
	aws s3 sync data/interim s3://$(BUCKET)/data/interim --profile $(PROFILE)
	aws s3 sync data/processed s3://$(BUCKET)/data/processed --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/raw data/raw
	aws s3 sync s3://$(BUCKET)/data/interim data/interim
	aws s3 sync s3://$(BUCKET)/data/processed data/processed
else
	aws s3 sync s3://$(BUCKET)/data/raw data/raw --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)/data/interim data/interim --profile $(PROFILE)
	aws s3 sync s3://$(BUCKET)/data/processed data/processed --profile $(PROFILE)
endif

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
