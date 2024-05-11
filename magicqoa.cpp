#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "stdint.h"
#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <netinet/in.h>
#include <stdint.h>
#include <expected>

#include <iostream>
#include <array>
#include <vector>


#include <arpa/inet.h>

namespace magicqoa
{

using u64 = uint64_t;
using u32 = uint32_t;
using u16 = uint16_t;
using i16 = int16_t;
using u8 = uint8_t;
using sz = size_t;
using namespace std;

constexpr size_t QOA_SLICE_SAMPLES = 20;


struct FileReader
{
    [[nodiscard("How much data read must be check")]] virtual size_t read(void *buf, size_t size) = 0;
    virtual bool seek(size_t pos) { return false; }
    virtual size_t tell() { return 0; }

    expected<u32, string>  read_u32()
    {
        u32 val;
        if (read(&val, sizeof(val)) != sizeof(val))
            return std::unexpected("Failed to read u32");
        return htonl(val);
    }

    expected<u16, string>  read_u16()
    {
        u16 val;
        if (read(&val, sizeof(val)) != sizeof(val))
            return std::unexpected("Failed to read u16");
        return htons(val);
    }

    expected<u8, string>  read_u8()
    {
        u8 val;
        if (read(&val, sizeof(val)) != sizeof(val))
            return std::unexpected("Failed to read u8");
        return val;
    }

    expected<u32, string>  read_u24()
    {
        u32 val;
        if (read(&val, 3) != 3)
            return std::unexpected("Failed to read u24");
        val = htonl(val);
        return val >> 8;
    }

    expected<void, string>  read_bytes(void *buf, size_t size)
    {
        if (read(buf, size) != size)
            return std::unexpected("Failed to read bytes");
        return {};
    }
};

struct StandardFileReader : FileReader
{
    std::istream &stream;
    StandardFileReader(std::istream &stream) : stream(stream) {}

    size_t read(void *buf, size_t size) override
    {
        stream.read((char *)buf, size);
        return stream.gcount();
    }

    bool seek(size_t pos) override
    {
        stream.seekg(pos);
        return stream.good();
    }

    size_t tell() override
    {
        return stream.tellg();
    }
};

struct FileHeader
{
    char magic[4];    // "qoaf"
    uint32_t samples; // samples per channel

    static expected<FileHeader, string> read(FileReader &reader)
    {
        FileHeader header;
        size_t n = reader.read(header.magic, sizeof(header.magic));
        if (n != 4 || memcmp(header.magic, "qoaf", 4) != 0)
            return std::unexpected("Invalid magic");

        auto samples = reader.read_u32();
        if(!samples.has_value())
            return std::unexpected("Failed to read samples");
        header.samples = samples.value();
        return header;
    }
};

struct FrameHeader
{
    uint8_t num_channels;
    uint32_t sample_rate; // Stored as 24bit!
    uint16_t frame_samples; // number of samples in this frame
    uint16_t frame_size;   // size of the frame in bytes (including this header)

    static expected<FrameHeader, string> read(FileReader &reader)
    {
        FrameHeader header;
        auto num_channels = reader.read_u8();
        if(!num_channels.has_value())
            return std::unexpected("Failed to read num_channels");
            if(num_channels.value() == 0)
                return std::unexpected("num_channels cannot be 0");
        header.num_channels = num_channels.value();

        auto sample_rate = reader.read_u24();
        if(!sample_rate.has_value())
            return std::unexpected("Failed to read sample_rate");
        if(sample_rate.value() == 0)
            return std::unexpected("sample_rate cannot be 0");
        header.sample_rate = sample_rate.value();

        auto frame_samples = reader.read_u16();
        if(!frame_samples.has_value())
            return std::unexpected("Failed to read frame_samples");
        if(frame_samples.value() == 0)
            return std::unexpected("frame_samples cannot be 0");
        header.frame_samples = frame_samples.value();

        auto frame_size = reader.read_u16();
        if(!frame_size.has_value())
            return std::unexpected("Failed to read frame_size");
        if(frame_size.value() == 0)
            return std::unexpected("frame_size cannot be 0");
        header.frame_size = frame_size.value();

        return header;
    }
};

struct LSMState
{
    array<i16, 4> history;
    array<i16, 4> weights;

    static expected<LSMState, string> read(FileReader &reader)
    {
        LSMState state;
        for(auto& history : state.history) {
            auto val = reader.read_u16();
            if(!val.has_value())
                return std::unexpected("Failed to read history");
            history = val.value();
        }

        for(auto& weight : state.weights) {
            auto val = reader.read_u16();
            if(!val.has_value())
                return std::unexpected("Failed to read weights");
            weight = val.value();
        }

        return state;
    }
};

const static float dequant_tab[8] = {0.75, -0.75, 2.5, -2.5, 4.5, -4.5, 7, -7};
const static int sf_dequant_tab[16] = {
    (int)round(pow(1, 2.75)),
    (int)round(pow(2, 2.75)),
    (int)round(pow(3, 2.75)),
    (int)round(pow(4, 2.75)),
    (int)round(pow(5, 2.75)),
    (int)round(pow(6, 2.75)),
    (int)round(pow(7, 2.75)),
    (int)round(pow(8, 2.75)),
    (int)round(pow(9, 2.75)),
    (int)round(pow(10, 2.75)),
    (int)round(pow(11, 2.75)),
    (int)round(pow(12, 2.75)),
    (int)round(pow(13, 2.75)),
    (int)round(pow(14, 2.75)),
    (int)round(pow(15, 2.75)),
    (int)round(pow(16, 2.75))
};

struct slice
{
    u8 quant;
    std::array<u8, QOA_SLICE_SAMPLES> qr;

    static expected<slice, string> read(FileReader &reader)
    {
        u64 data = 0;
        sz n = reader.read(&data, 8);
        if(n != 8)
            return std::unexpected("Failed to read slice");
        data = be64toh(data);

        slice s;
        // ┌─ qoa_slice_t ── 64 bits, 20 samples ────────────/ /────────────┐
        // |     Byte[0]     |               Byte[1]         \ \   Byte[7]  |
        // | 7  6  5  4  3  2  1  0 | 7  6  5  4  3  2  1  0 / /   2  1  0  |
        // ├────────────┼────────┼──┴─────┼────────┼─────────\ \──┼─────────┤
        // |  sf_quant  │  qr00  │  qr01  │  qr02  │   qr03  / /  │   qr19  |
        // └────────────┴────────┴────────┴────────┴─────────\ \──┴─────────┘

        s.quant = data >> 60;
        // data = data & ~(0xfUL << 56); // shoudn't be needed
        for (int i = s.qr.size() - 1; i >= 0; i--) {
            s.qr[i] = data & 0x7;
            data >>= 3;
        }
        return s;
    }

    void decode(LSMState& state, i16* ptr, size_t stride) const
    {
        int sf = sf_dequant_tab[quant];
        for(int n = 0; n < qr.size(); n++) {
            float fr = sf * dequant_tab[qr[n]];
            int r = fr < 0 ? ceil(fr - 0.5) : floor(fr + 0.5);

            int p = 0;
            for(int i=0; i<4; i++) {
                p += state.history[i] * state.weights[i];
            }
            p >>= 13;

            int sample = r + p;
            if(sample > SHRT_MAX)
                sample = SHRT_MAX;
            else if(sample < SHRT_MIN)
                sample = SHRT_MIN;

            int delta = r >> 4;
            for (int i=0; i<4; i++)
                state.weights[i] += (state.history[i] < 0) ? -delta : delta;
            for (int i=0; i<3; i++)
                state.history[i] = state.history[i+1];
            state.history[3] = sample;

            ptr[n * stride] = sample;
        }
    }
};


struct Frame
{
    FrameHeader header;
    vector<LSMState> states;
    vector<slice> slices;

    static expected<Frame, string> read(FileReader &reader)
    {
        Frame frame;
        auto header = FrameHeader::read(reader);
        if(!header.has_value())
            return std::unexpected("Failed to read header: " + header.error()); 
        frame.header = header.value();

        for (int i = 0; i < frame.header.num_channels; i++) {
            auto state = LSMState::read(reader);
            if(!state.has_value())
                return std::unexpected("Failed to read state");
            frame.states.push_back(state.value());
        }

        for (int i = 0; i < 256 * frame.header.num_channels; i++) {
            auto slice = slice::read(reader);
            if(!slice.has_value())
                break;
            frame.slices.push_back(slice.value());
        }
        if(frame.slices.size() % frame.header.num_channels != 0)
            return std::unexpected("Unexpected end of file");

        return frame;
    }

    std::vector<i16> decode()
    {
        std::vector<i16> samples;
        samples.resize(slices.size() * 20);

        for (int sid = 0; sid < slices.size(); sid++)
        {
            auto channel = sid % header.num_channels;
            auto occurrence = sid / header.num_channels;
            const auto& slice = slices[sid];
            auto &state = states[channel];
            slice.decode(state, samples.data() + occurrence * header.num_channels * QOA_SLICE_SAMPLES + channel, header.num_channels);
        }
        return samples;
    }
};

enum class StreamDecoderReadState
{
    ReadFrameHeader,
    ReadLSMState,
    ReadSlice,
};

struct QOAPlayer
{
    FileReader &reader;

    QOAPlayer(FileReader &reader) : reader(reader) {
        // There must be enough data to read the header
        auto head = FileHeader::read(reader);
        if(!head.has_value())
            throw std::runtime_error("Failed to read header: " + head.error());
        header = head.value();

        auto frame_head = FrameHeader::read(reader);
        if(!frame_head.has_value())
            throw std::runtime_error("Failed to read the first frame header: " + frame_head.error());

        frame_header = frame_head.value();
        read_state = StreamDecoderReadState::ReadLSMState;
        lsm_read_idx = 0;
        num_channels = frame_header.num_channels;
        decode_tmp.resize(frame_header.num_channels * QOA_SLICE_SAMPLES);
    }

    bool isStream() const { return header.samples == 0; }
    int channels() const { return num_channels; }

    /*
     * Decode as much as possible into the buffer.
     * @param buffer: buffer to decode into
     * @param buf_size_elements: size of the buffer in elements
     * @return number of PCM samples decoded. -1 means there is no more data to decode.
    **/
    ssize_t decode(i16* buffer, size_t buf_size_elements)
    {
        if(buf_size_elements == 0)
            return 0;

        size_t insert_pos = 0;
        if(buffered_pcm.size() > 0) {
            assert(buffered_pcm.size() % num_channels == 0);
            size_t can_copy = std::min(buf_size_elements, buffered_pcm.size());
            memcpy(buffer, buffered_pcm.data(), can_copy * sizeof(i16));
            buffered_pcm.erase(buffered_pcm.begin(), buffered_pcm.begin() + can_copy);
            insert_pos += can_copy;
        }

        if(insert_pos == buf_size_elements)
            return insert_pos;
        while(true) {
            if(read_state == StreamDecoderReadState::ReadFrameHeader) {
                auto frame_head = FrameHeader::read(reader);
                if(!frame_head.has_value())
                    return insert_pos;

                frame_header = frame_head.value();
                read_state = StreamDecoderReadState::ReadLSMState;
                lsm_read_idx = 0;
            }
            else if(read_state == StreamDecoderReadState::ReadLSMState) {
                states.resize(frame_header.num_channels);
                for (; lsm_read_idx < frame_header.num_channels; lsm_read_idx++) {
                    auto state = LSMState::read(reader);
                    if(!state.has_value())
                        return insert_pos;
                    states[lsm_read_idx] = state.value();
                }
                slice_read_idx = 0;
                decoded_sample_ch = 0;
                read_state = StreamDecoderReadState::ReadSlice;
            }
            else if(read_state == StreamDecoderReadState::ReadSlice) {
                while(slice_read_idx < 256 * frame_header.num_channels && slices.size() != frame_header.num_channels) {
                    auto slice = slice::read(reader);
                    if(!slice.has_value())
                        break;
                    slices.push_back(slice.value());
                    slice_read_idx++;
                }
                decode_tmp.resize(frame_header.num_channels * QOA_SLICE_SAMPLES);
                for (int channel = 0; channel < frame_header.num_channels; channel++) {
                    const auto& slice = slices[channel];
                    auto &state = states[channel];
                    slice.decode(state, decode_tmp.data() + channel, frame_header.num_channels);
                }
                slices.clear();
                size_t buf_size_aligned_channels = (buf_size_elements / frame_header.num_channels) * frame_header.num_channels;
                if(buf_size_aligned_channels == 0)
                    return 0;
                size_t can_copy = min(min(buf_size_aligned_channels - insert_pos, decode_tmp.size()), (header.samples - decoded_sample_ch) * frame_header.num_channels);
                memcpy(buffer + insert_pos, decode_tmp.data(), can_copy * sizeof(i16));
                insert_pos += can_copy;
                decoded_sample_ch += can_copy / frame_header.num_channels;
                if(slice_read_idx == 256 * frame_header.num_channels || decoded_sample_ch == header.samples) {
                    slice_read_idx = 0;
                    read_state = StreamDecoderReadState::ReadFrameHeader;
                }
                if(insert_pos == buf_size_aligned_channels) {
                    buffered_pcm.insert(buffered_pcm.end(), decode_tmp.begin() + can_copy, decode_tmp.end());
                    break;
                }
            }
        }
        return insert_pos;
    }

    FileHeader header;

    FrameHeader frame_header;
    std::vector<i16> buffered_pcm;
    std::vector<LSMState> states;

    int num_channels = 0;

    // Tracks what we need to read next
    StreamDecoderReadState read_state = StreamDecoderReadState::ReadFrameHeader;
    size_t lsm_read_idx = 0;
    size_t slice_read_idx = 0;
    size_t decoded_sample_ch = 0;

    // TOOD: Optimize and remove these
    std::vector<slice> slices;
    vector<i16> decode_tmp;
};
}

int main()
{
    // std::string filename = "../samples/bandcamp/qoa/allegaeon-beasts-and-worms.qoa";
    std::string filename = "天ノ弱.qoa";
    std::ifstream file(filename, std::ios::binary);
    magicqoa::StandardFileReader reader(file);

    if(!file.is_open()) {
        printf("Failed to open file\n");
        return 1;
    }

#if 1
    magicqoa::QOAPlayer player(reader);
    std::vector<int16_t> pcm;
    std::vector<int16_t> buffer(1024);
    while(true) {
        auto decoded = player.decode(buffer.data(), buffer.size());
        if(decoded == 0 || decoded == -1) {
            std::cout << "Decode ended before the length specsified by file header";
            break;
        }
        if(pcm.size() / player.channels() == player.header.samples)
            break;
        pcm.insert(pcm.end(), buffer.begin(), buffer.begin() + decoded);
    }

    std::ofstream out("out.pcm", std::ios::binary);
    out.write((char *)pcm.data(), pcm.size() * sizeof(uint16_t));
    out.close();
#else
    auto header = magicqoa::FileHeader::read(reader);

    std::cout << "Samples per channel: " << std::hex << header->samples << std::endl;

    size_t frame_count = 0;
    std::vector<uint16_t> pcm = {};
    size_t decoded_channel_samples = 0;
    while (true)
    {
        auto frame = magicqoa::Frame::read(reader);
        std::cout << "Frame " << frame_count++ << std::endl;
        if(!frame.has_value()) {
            std::cout << "Message: " << frame.error() << std::endl;
            break;
        }
        std::cout << "  Num channels: " << std::dec << (int)frame->header.num_channels << std::endl;
        std::cout << "  Sample rate: " << std::dec << frame->header.sample_rate << std::endl;
        std::cout << "  Frame samples: " << std::dec << frame->header.frame_samples << std::endl;

        auto samples = frame->decode();
        std::cout << "Decoded " << samples.size() << " samples\n";
        pcm.insert(pcm.end(), samples.begin(), samples.end());
        decoded_channel_samples += frame->header.frame_samples;
        if(decoded_channel_samples >= header->samples && header->samples != 0)
            break;
    }

    std::ofstream out("out.pcm", std::ios::binary);
    out.write((char *)pcm.data(), pcm.size() * sizeof(uint16_t));
    out.close();
#endif


}
