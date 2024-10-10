# VideoEditor

Projeto criado para ajudar um amigo na atividade de cortar vídeos apartir do ponto onde uma frase é encontrada. Para realizar o projeto, utilizei duas maneiras para transcrever o áudio.

* Speech_recognition: Uma biblioteca popular em Python que facilita a conversão de fala em texto. Pode ser integrada com APIs como Google Web Speech API e Microsoft Azure Speech.
* transformers: São uma arquitetura de rede neural introduzida em 2017 no artigo "Attention is All You Need" por pesquisadores da Google. Eles revolucionaram o campo do processamento de linguagem natural (NLP) e de outras tarefas envolvendo séries temporais e processamento de sequências, como tradução automática, geração de texto, síntese de fala e até visão computacional. A característica principal dos transformers é o uso intensivo do mecanismo de atenção para processar dados sequenciais, superando limitações de arquiteturas anteriores, como as redes recorrentes (RNNs) e LSTMs.

O modelo utilizado no transformers foi o "openai/whisper-large-v3" que se mostrou uma opção execelente para realizar as tarefas. O mesmo é mais lento que o speech_recognition, porém é mais preciso.

O projeto ainda está em construção de algumas melhorias. Sendo assim se for utilizar, por favor, olhar bem as funções e ver se atende o que você precisa. Dentro de algumas dias eu devo trazer as atualizações para melhorar o projeto.