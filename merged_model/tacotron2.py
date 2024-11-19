import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super().__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = nn.Conv1d(2, attention_n_filters, 
                                     kernel_size=attention_kernel_size,
                                     padding=padding, bias=False)
        self.location_dense = nn.Linear(attention_n_filters, attention_dim, bias=False)

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super().__init__()
        self.query_layer = nn.Linear(attention_rnn_dim, attention_dim, bias=False)
        self.memory_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                          attention_location_kernel_size,
                                          attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                             attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_memory + processed_attention_weights))
        return energies.squeeze(-1)

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x

class Postnet(nn.Module):
    def __init__(self, n_mel_channels, postnet_embedding_dim,
                 postnet_kernel_size, postnet_n_convolutions):
        super().__init__()
        
        self.convolutions = nn.ModuleList()
        
        # 첫 번째 컨볼루션 레이어
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Tanh(),
                nn.Dropout(0.5)
            )
        )

        # 중간 컨볼루션 레이어들
        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(postnet_embedding_dim, postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=(postnet_kernel_size - 1) // 2),
                    nn.BatchNorm1d(postnet_embedding_dim),
                    nn.Tanh(),
                    nn.Dropout(0.5)
                )
            )

        # 마지막 컨볼루션 레이어
        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=(postnet_kernel_size - 1) // 2),
                nn.BatchNorm1d(n_mel_channels),
                nn.Dropout(0.5)
            )
        )

    def forward(self, x):
        """
        Args:
            x: 멜 스펙트로그램 [batch_size, n_mel_channels, time]
        Returns:
            수정된 멜 스펙트로그램
        """
        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)
        x = self.convolutions[-1](x)
        return x

class Encoder(nn.Module):
    def __init__(self, encoder_embedding_dim, encoder_n_convolutions,
                 encoder_kernel_size, encoder_lstm_dim):
        super().__init__()

        # 3개의 conv layers
        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                nn.Conv1d(encoder_embedding_dim,
                         encoder_embedding_dim,
                         kernel_size=encoder_kernel_size,
                         stride=1,
                         padding=int((encoder_kernel_size - 1) / 2)),
                nn.BatchNorm1d(encoder_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.5))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # Bi-directional LSTM (128 * 2 = 256)
        self.lstm = nn.LSTM(encoder_embedding_dim,
                           encoder_lstm_dim,
                           1,
                           batch_first=True,
                           bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

class Decoder(nn.Module):
    def __init__(self, n_mel_channels, encoder_embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 attention_rnn_dim, decoder_rnn_dim, prenet_dim,
                 max_decoder_steps, gate_threshold, p_attention_dropout,
                 p_decoder_dropout):
        super().__init__()
        
        # 파라미터 저장
        self.n_mel_channels = n_mel_channels
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_dim = attention_dim
        self.attention_location_n_filters = attention_location_n_filters
        self.attention_location_kernel_size = attention_location_kernel_size
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        
        # Prenet
        self.prenet = Prenet(
            n_mel_channels,
            [prenet_dim, prenet_dim]
        )
        
        # Attention RNN
        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim
        )
        
        # Attention Layer
        self.attention_layer = Attention(
            attention_rnn_dim,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size
        )
        
        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim
        )
        
        # Linear Projection
        self.linear_projection = nn.Linear(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels
        )
        
        # Gate Layer
        self.gate_layer = nn.Linear(
            decoder_rnn_dim + encoder_embedding_dim,
            1
        )

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.parse_decoder_inputs(decoder_inputs)
        decoder_outputs = self.decode(decoder_input, memory, memory_lengths)
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            decoder_outputs)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, memory, memory_lengths=None):
        """ Decoder forward pass for training
        PARAMS
        ------
        decoder_input: list of mel frames
        memory: Encoder outputs
        memory_lengths: Encoder output lengths for attention masking.
        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.parse_decoder_inputs(decoder_input)
        decoder_outputs = []
        attention_weights = []
        attention_weights_cum = []
        attention_context = []

        # Initialize decoder states
        attention_hidden = torch.zeros(
            decoder_input.size(0), self.attention_rnn_dim, device=decoder_input.device)
        attention_cell = torch.zeros(
            decoder_input.size(0), self.attention_rnn_dim, device=decoder_input.device)
        decoder_hidden = torch.zeros(
            decoder_input.size(0), self.decoder_rnn_dim, device=decoder_input.device)
        decoder_cell = torch.zeros(
            decoder_input.size(0), self.decoder_rnn_dim, device=decoder_input.device)
        attention_weights = torch.zeros(
            decoder_input.size(0), memory.size(1), device=decoder_input.device)
        attention_weights_cum = torch.zeros(
            decoder_input.size(0), memory.size(1), device=decoder_input.device)

        # Initialize previous context
        attention_context = torch.zeros(
            decoder_input.size(0), self.encoder_embedding_dim, device=decoder_input.device)

        # Time first: (batch_size, time_steps, n_mel_channels)
        processed_memory = self.memory_layer(memory)
        if memory_lengths is not None:
            mask = ~get_mask_from_lengths(memory_lengths)
        else:
            mask = None

        for di in range(decoder_input.size(1)):
            decoder_input_di = decoder_input[:, di]
            mel_output, gate_output, attention_weights = self.decode_step(
                decoder_input_di, memory, processed_memory, attention_hidden,
                attention_cell, decoder_hidden, decoder_cell, attention_weights,
                attention_weights_cum, attention_context, mask)

            decoder_outputs += [mel_output]
            gate_outputs += [gate_output]
            alignments += [attention_weights]

            # Update previous context
            attention_context = self.attention_layer(
                attention_hidden, memory, processed_memory,
                attention_weights.unsqueeze(1), mask)[0]

        decoder_outputs = torch.stack(decoder_outputs).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        alignments = torch.stack(alignments).transpose(0, 1)

        return decoder_outputs, gate_outputs, alignments

class Tacotron2(nn.Module):
    # 고정된 값들을 클래스 변수로 정의
    attention_location_n_filters = 32
    attention_location_kernel_size = 31
    decoder_rnn_dim = 512
    prenet_dim = 128
    max_decoder_steps = 1000
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1
    postnet_embedding_dim = 256
    postnet_kernel_size = 5

    def __init__(self, n_mel_channels, vocab_size, embedding_dim, 
                 encoder_n_convolutions, encoder_kernel_size,
                 attention_rnn_dim, attention_dim):
        super().__init__()
        
        # 임베딩 레이어 (256차원)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 인코더
        self.encoder = Encoder(
            encoder_embedding_dim=embedding_dim,  # 256
            encoder_n_convolutions=encoder_n_convolutions,
            encoder_kernel_size=encoder_kernel_size,
            encoder_lstm_dim=int(embedding_dim/2)  # 128 (bi-directional = 256)
        )
        
        # 디코더
        self.decoder = Decoder(
            n_mel_channels=n_mel_channels,
            encoder_embedding_dim=embedding_dim,  # 256
            attention_dim=attention_dim,  # 128
            attention_location_n_filters=self.attention_location_n_filters,
            attention_location_kernel_size=self.attention_location_kernel_size,
            attention_rnn_dim=attention_rnn_dim,  # 512
            decoder_rnn_dim=self.decoder_rnn_dim,  # 512
            prenet_dim=self.prenet_dim,  # 128
            max_decoder_steps=self.max_decoder_steps,
            gate_threshold=self.gate_threshold,
            p_attention_dropout=self.p_attention_dropout,
            p_decoder_dropout=self.p_decoder_dropout
        )
        
        # 포스트넷
        self.postnet = Postnet(
            n_mel_channels=n_mel_channels,
            postnet_embedding_dim=self.postnet_embedding_dim,
            postnet_kernel_size=self.postnet_kernel_size,
            postnet_n_convolutions=5
        )

    def forward(self, text_inputs, text_lengths, mel_inputs, mel_lengths):
        """
        텍스트를 멜 스펙트로그램으로 변환
        """
        # 텍스트 임베딩
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        
        # 인코더 통과
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        # 디코더 통과
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_inputs, text_lengths)
        
        # 포스트넷 통과
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return mel_outputs_postnet, mel_outputs, gate_outputs, alignments

    def inference(self, text_inputs):
        """
        추론 시 사용되는 forward 메서드
        """
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return mel_outputs_postnet, mel_outputs, gate_outputs, alignments

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask 