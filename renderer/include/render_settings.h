#pragma once

namespace render
{
    struct Color
    {
        double rgb[3];
    };

    inline constexpr int windowWidth = 1920;
    inline constexpr int windowHeight = 1080;

    inline Color black{ .rgb = { 0.0, 0.0, 0.0 } };
    inline Color red{ .rgb = { 1.0, 0.0, 0.0 } };
    inline Color green{ .rgb = { 0.0, 1.0, 0.0 } };
    inline Color blue{ .rgb = { 0.0, 0.0, 1.0 } };
    inline Color brown{ .rgb = { 0.8, 0.48, 0.25 } };
    inline Color purple{ .rgb = { 0.5, 0.0, 0.5 } };   // Purple color definition
    inline Color orange{ .rgb = { 1.0, 0.5, 0.0 } };   // Orange color definition
    inline Color yellow{ .rgb = { 1.0, 1.0, 0.0 } };   // Yellow color definition
    inline Color cyan{ .rgb = { 0.0, 1.0, 1.0 } };     // Cyan color definition
    inline Color magenta{ .rgb = { 1.0, 0.0, 1.0 } };  // Magenta color definition
    inline Color lime{ .rgb = { 0.0, 1.0, 0.0 } };
}