#pragma once

namespace outerNamespace
{

    namespace innerNamespace
    {
    
        void func0()
        {
            // do something
        }

        

        class A
        {
            uint x;
            int func1(int a, float b)
            {
                func0();
                return 0;
            }   
        }

        struct B
        {
            int y;
            A a;
            void func2()
            {
                a.func1(10, 2.5f);
            }

            void func3();


        }

        enum E
        {
            VAL1,
            VAL2,
            VAL3
        };

        void foo()
        {
            outerNamespace::innerNamespace::B::func3();
        }
    }
}
