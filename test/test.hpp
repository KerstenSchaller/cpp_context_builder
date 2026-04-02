#pragma once
namespace Z{
namespace X
{
    void func1()
    {}

    class A
    {
    public:
        A(int x)
        {}
        void classFunc(int x, float y)
        {}
    };
}
}



   

        void callingFunc()
        {
            Z::X::func1();
        }

        int callingClass()
        {
            Z::X::A a(42);
            a.classFunc(42, 3.14f);
            return 42;
        }


