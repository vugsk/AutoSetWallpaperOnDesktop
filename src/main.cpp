
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>


static std::vector<std::filesystem::path> generate_files(const std::string& path)
{
    std::filesystem::directory_iterator dir{path};
    std::vector<std::filesystem::path> arr(begin(dir), end(dir));
    std::ranges::sort(arr);
    uint32_t count = 0;
    for (const auto& i : arr)
    {
        std::string f = i.parent_path().generic_string() + '/' +
                std::to_string(count++) + i.extension().generic_string();
        rename(i.c_str(), f.c_str());
    }
    return arr;
}

static uint32_t random_number(const uint32_t min, const uint32_t max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    return std::uniform_int_distribution<>(
        static_cast<int>(min), static_cast<int>(max))(gen);
}

static std::filesystem::path find_file(const std::vector<std::filesystem::path>& files)
{
    const auto file_path = std::ranges::find_if(files,
        [ran = random_number(0, files.size() - 1)]
                (const std::filesystem::path& i) -> bool
    {
        return stoul(i.filename().string().substr(0, i.filename().string().find(i.extension().string()))) == ran;
    });

    if (file_path != files.end())
        return files[static_cast<uint64_t>(file_path - files.begin())];
    return {};
}

using namespace cv;

static constexpr char hex[] = {
    '0','1','2','3','4','5','6','7',
    '8','9','A','B','C','D','E','F'
};

template<typename T>
static constexpr char convertHex(T num)
{
    for (uint8_t i = 0; i < 16; i++)
        if (i == num)
            return hex[i];
    return '\0';
}

template<typename T>
static std::string decToHex(T number)
{
    if (number < 16)
    {
        if (number < 10)
            return std::string{'0', convertHex(number)};
        return std::string{convertHex(number)};
    }

    T ui = 0;
    std::string it;
    do {
        ui = number / 16;
        if (ui == 0) break;
        it.push_back(convertHex(ui));
        number = number - (ui * 16);
        it.push_back(convertHex(number));
    } while (ui != 0);
    return it;
}

// rewrite code
static std::vector<std::string> getPalitryRGB(const std::string& path)
{
    const Mat image = imread(path);
    if (image.empty())
        return std::vector<std::string>{};

    // Преобразование изображения в формат, подходящий для K-means
    Mat data;
    image.convertTo(data, CV_32F); // Преобразуем в тип float
    data = data.reshape(1, image.rows * image.cols); // Преобразуем в одномерный массив пикселей

    // Параметры для K-means
    static constexpr auto K = 5; // Количество кластеров (основных цветов)
    Mat labels, centers;
    const TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 1.0); // Критерий остановки
    kmeans(data, K, labels, criteria, 3,
        KMEANS_PP_CENTERS, centers); // Выполняем K-means

    // Преобразование центров кластеров обратно в формат BGR
    centers = centers.reshape(3, centers.rows); // Преобразуем центры в формат (K, 3)
    centers.convertTo(centers, CV_8U); // Преобразуем в тип uchar

    // Вывод основных цветов
    std::vector<std::string> vec;
    for (auto i = 0; i < K; i++) {
        const auto color = centers.at<Vec3b>(i);
        vec.push_back('#' + decToHex(static_cast<int32_t>(color[2]))
                            + decToHex(static_cast<int32_t>(color[1]))
                            + decToHex(static_cast<int32_t>(color[0])));
    }
    return vec;
}

static bool emptyFile(std::fstream& file)
{
    if (!file.is_open())
        return true;
    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    return size == 0;
}

template<typename T, typename F>
static nlohmann::json fileWR(const std::string& path, T&& f, F&& func)
{
    std::fstream file(path, f);
    nlohmann::json data = func(file);
    file.close();
    return data;
}

nlohmann::json hj(const std::string& path)
{
    return fileWR(path, std::ios::in,
        [](std::fstream& file)
    {
        if (!file.is_open())
            return nlohmann::json{};
        return nlohmann::json::parse(file);
    });
}

int main()
{
    static constexpr auto dir_json_save = "config.json";
    nlohmann::json datar = hj(dir_json_save);

    const std::filesystem::path p = find_file(generate_files(datar["path_read_files_img"]));
    const std::string img_path = p.filename().generic_string().substr(0,
        p.filename().generic_string().find(p.extension().generic_string()));
    const std::string namefile_save_colors = std::string(datar["path_save_file_color"]) + "save_colors.json";

    const nlohmann::json k = fileWR(namefile_save_colors, std::ios::in,
        [&img_path](std::fstream& file)
    {
        if (!file.is_open())
            return nlohmann::json{};
        if (!emptyFile(file))
        {
            for (nlohmann::json kl = nlohmann::json::parse(file);
                const auto& [key, i] : kl.items())
            {
                if (key == img_path)
                    return i;
            }
        }
        return nlohmann::json{};
    });

    if (!k.empty())
    {
        std::cout << k << std::endl;
        return 0;
    }

    std::vector<std::string> vec = getPalitryRGB(p);

    nlohmann::json kl = hj(namefile_save_colors);
    fileWR(namefile_save_colors, std::ios::out,
        [&vec, &img_path, &kl](std::fstream& file)
    {
        auto func = [&vec, &file, &img_path](nlohmann::json& data)
        {
            data[img_path] = vec;
            file << data.dump(4);
        };

        if (!kl.empty())
        {
            auto is = false;
            for (const auto& [key, i] : kl.items())
                if (key == img_path)
                    is = true;
            if (!is)
                func(kl);
            return nlohmann::json{};
        }

        nlohmann::json fk;
        func(fk);
        return nlohmann::json{};
    });

    // do use colors in the system

    return 0;
}