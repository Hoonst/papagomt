
# RNN_CELL_TYPES=("rnn" "lstm" "gru")
# for RNN in ${RNN_CELL_TYPES[@]}; do
#     echo $RNN
#     python code/main.py\
#         --experiment-name='rnn-test'\
#         --root-path='data'\
#         --rnn-cell-type=$RNN\
#         --teacher-forcing-ratio=0.8\
#         --valid-ratio=0.2\
#         --epoch=10
# done

python code/main.py\
        --experiment-name='rnn-seq2seq'\
        --root-path='data'\
        --rnn-cell-type='rnn'\
        --teacher-forcing-ratio=0.8\
        --valid-ratio=0.2\
        --epoch=1\
        --use-attention=1\