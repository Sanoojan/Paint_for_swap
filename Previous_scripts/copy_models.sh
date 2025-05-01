
declare -a names=(  "checkpoits"
                    "Other_dependencies"
                    "pretrained"
                    "eval_tool"
                    )

declare -a names=( 
                    "intermediate_renact"
                    )


for name in "${names[@]}"
do
    scp -r sanoojan@10.127.30.114:/share/data/drive_3/Sanoojan/needed/Paint_for_swap/${name} ${name}
done
