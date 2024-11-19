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
        
        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                nn.Conv1d(encoder_embedding_dim, encoder_embedding_dim,
                         encoder_kernel_size, stride=1,
                         padding=int((encoder_kernel_size - 1) / 2)),
                nn.BatchNorm1d(encoder_embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(encoder_embedding_dim, encoder_lstm_dim,
                           num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        """
        x: [B, embed_dim, T]
        input_lengths: [B]
        """
        # Conv layers
        for conv in self.convolutions:
            x = conv(x)
        
        # Prepare for LSTM
        x = x.transpose(1, 2)  # [B, T, embed_dim]
        
        # Pack sequence
        input_lengths = input_lengths.cpu()  # lengths를 CPU로 이동
        
        # Sort by length for packing
        input_lengths, sort_idx = torch.sort(input_lengths, descending=True)
        x = x[sort_idx]
        
        # Pack the sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths.cpu(), batch_first=True)
        
        # LSTM forward
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x_packed)
        
        # Unpack the sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        
        # Restore original order
        _, unsort_idx = torch.sort(sort_idx)
        outputs = outputs[unsort_idx]
        
        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = conv(x)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs

class Decoder(nn.Module):
    def __init__(self, n_mel_channels, encoder_embedding_dim,
                 attention_dim, attention_location_n_filters,
                 attention_location_kernel_size, attention_rnn_dim,
                 decoder_rnn_dim, prenet_dim, max_decoder_steps,
                 gate_threshold, p_attention_dropout, p_decoder_dropout):
        super().__init__()
        
        self.n_mel_channels = n_mel_channels
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

        # Prenet
        self.prenet = Prenet(n_mel_channels, [prenet_dim, prenet_dim])

        # Attention RNN
        self.attention_rnn = nn.LSTMCell(
            prenet_dim + encoder_embedding_dim,
            attention_rnn_dim)

        # Attention Layer
        self.attention_layer = Attention(
            attention_rnn_dim, encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)

        # Decoder RNN
        self.decoder_rnn = nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim,
            decoder_rnn_dim)

        # Linear Projection
        self.linear_projection = nn.Linear(
            decoder_rnn_dim + encoder_embedding_dim,
            n_mel_channels)  # 출력을 n_mel_channels로 수정

        # Gate Layer
        self.gate_layer = nn.Linear(
            decoder_rnn_dim + encoder_embedding_dim, 1,
            bias=True)

        # Attention 관련 레이어 추가
        self.attention_layer = Attention(
            attention_rnn_dim,
            encoder_embedding_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size
        )
        
        # Memory layer 추가
        self.memory_layer = nn.Linear(
            encoder_embedding_dim,
            attention_dim,
            bias=False
        )
        
        # Attention context projection
        self.attention_projection = nn.Linear(
            encoder_embedding_dim,
            decoder_rnn_dim,
            bias=False
        )

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        Args:
            decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        """
        # (B, n_mel_channels, T) -> (B, T, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.contiguous()
        
        # (B, T, n_mel_channels) -> (T, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        Args:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            alignments: alignments from the decoder
        """
        # (T, B, n_mel_channels) -> (B, T, n_mel_channels)
        mel_outputs = mel_outputs.transpose(0, 1).contiguous()
        
        # (T, B) -> (B, T)
        gate_outputs = gate_outputs.transpose(0, 1).contiguous()
        
        # (T, B, n_text) -> (B, T, n_text)
        alignments = alignments.transpose(0, 1).contiguous()

        return mel_outputs, gate_outputs, alignments

    def forward(self, encoder_outputs, decoder_inputs, memory_lengths=None):
        """
        Args:
            encoder_outputs: [batch_size, max_time, encoder_embedding_dim]
            decoder_inputs: [batch_size, n_mel_channels, max_time]
            memory_lengths: [batch_size]
        """
        # decoder_inputs 형태 변환
        decoder_inputs = decoder_inputs.transpose(1, 2)  # [batch_size, max_time, n_mel_channels]
        
        # 초기 상태 초기화
        batch_size = encoder_outputs.size(0)
        max_time = decoder_inputs.size(1)
        
        # 초기 attention context
        attention_context = torch.zeros(
            batch_size,
            self.encoder_embedding_dim
        ).to(encoder_outputs.device)
        
        # 초기 attention hidden states
        attention_hidden = torch.zeros(
            batch_size,
            self.attention_rnn_dim
        ).to(encoder_outputs.device)
        
        attention_cell = torch.zeros(
            batch_size,
            self.attention_rnn_dim
        ).to(encoder_outputs.device)
        
        # 초기 decoder states
        decoder_hidden = torch.zeros(
            batch_size,
            self.decoder_rnn_dim
        ).to(encoder_outputs.device)
        
        decoder_cell = torch.zeros(
            batch_size,
            self.decoder_rnn_dim
        ).to(encoder_outputs.device)
        
        # 초기 attention weights
        attention_weights = torch.zeros(
            batch_size,
            encoder_outputs.size(1)
        ).to(encoder_outputs.device)
        
        # 출력을 저장할 리스트
        mel_outputs, gate_outputs, alignments = [], [], []
        
        # Memory를 미리 처리
        processed_memory = self.memory_layer(encoder_outputs)
        
        # 각 타임스텝에 대해 처리
        for i in range(max_time):
            current_input = decoder_inputs[:, i, :]  # [batch_size, n_mel_channels]
            current_input = self.prenet(current_input)  # [batch_size, prenet_dim]
            
            # Attention RNN
            cell_input = torch.cat((current_input, attention_context), -1)
            attention_hidden, attention_cell = self.attention_rnn(
                cell_input, (attention_hidden, attention_cell))
            attention_hidden = F.dropout(
                attention_hidden, self.p_attention_dropout, self.training)
            
            # Attention 계산
            attention_weights_cat = torch.cat(
                (attention_weights.unsqueeze(1),
                 attention_weights.unsqueeze(1)),
                dim=1)
            attention_context, attention_weights = self.attention_layer(
                attention_hidden, encoder_outputs,
                processed_memory, attention_weights_cat,
                mask=None if memory_lengths is None else ~get_mask_from_lengths(memory_lengths))
            
            # Decoder RNN
            decoder_input = torch.cat((attention_hidden, attention_context), -1)
            decoder_hidden, decoder_cell = self.decoder_rnn(
                decoder_input, (decoder_hidden, decoder_cell))
            decoder_hidden = F.dropout(
                decoder_hidden, self.p_decoder_dropout, self.training)
            
            # Linear projection
            decoder_hidden_attention = torch.cat(
                (decoder_hidden, attention_context), dim=1)
            decoder_output = self.linear_projection(decoder_hidden_attention)
            gate_prediction = self.gate_layer(decoder_hidden_attention)
            
            # 결과 저장
            mel_outputs.append(decoder_output)
            gate_outputs.append(gate_prediction)
            alignments.append(attention_weights)
        
        # 리스트를 텐서로 변환
        mel_outputs = torch.stack(mel_outputs)
        gate_outputs = torch.stack(gate_outputs)
        alignments = torch.stack(alignments)
        
        # 출력 형식 변환
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        return mel_outputs, gate_outputs, alignments

class Tacotron2(nn.Module):
    attention_location_n_filters = 32
    attention_location_kernel_size = 31
    decoder_rnn_dim = 512
    prenet_dim = 256
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
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.encoder = Encoder(
            encoder_embedding_dim=embedding_dim,  # 256
            encoder_n_convolutions=encoder_n_convolutions,
            encoder_kernel_size=encoder_kernel_size,
            encoder_lstm_dim=int(embedding_dim/2)  # 128 (bi-directional = 256)
        )
        
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
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
        
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mel_inputs, text_lengths)
        
        # mel_outputs 차원 변환 (B, T, n_mel_channels) -> (B, n_mel_channels, T)
        mel_outputs_transpose = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = self.postnet(mel_outputs_transpose)
        mel_outputs_postnet = mel_outputs_transpose + mel_outputs_postnet
        
        # 결과를 다시 원래 형태로 변환 (B, n_mel_channels, T) -> (B, T, n_mel_channels)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        
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