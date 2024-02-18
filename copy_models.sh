
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
    scp -r sanoojan@10.127.30.114:/home/sanoojan/Paint_for_swap/${name} ${name}
done
