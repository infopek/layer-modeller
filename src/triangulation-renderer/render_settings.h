#pragma once

namespace render
{
    struct Color
    {
        double r;
        double g;
        double b;
        double a;
    };

    inline constexpr int windowWidth = 1920;
    inline constexpr int windowHeight = 1080;

    inline constexpr Color black{ 0.0, 0.0, 0.0, 1.0 };
    inline constexpr Color red{ 1.0, 0.2, 0.1, 0.4 };
}