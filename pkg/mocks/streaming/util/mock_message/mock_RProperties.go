// Code generated by mockery v2.46.0. DO NOT EDIT.

package mock_message

import mock "github.com/stretchr/testify/mock"

// MockRProperties is an autogenerated mock type for the RProperties type
type MockRProperties struct {
	mock.Mock
}

type MockRProperties_Expecter struct {
	mock *mock.Mock
}

func (_m *MockRProperties) EXPECT() *MockRProperties_Expecter {
	return &MockRProperties_Expecter{mock: &_m.Mock}
}

// Exist provides a mock function with given fields: key
func (_m *MockRProperties) Exist(key string) bool {
	ret := _m.Called(key)

	if len(ret) == 0 {
		panic("no return value specified for Exist")
	}

	var r0 bool
	if rf, ok := ret.Get(0).(func(string) bool); ok {
		r0 = rf(key)
	} else {
		r0 = ret.Get(0).(bool)
	}

	return r0
}

// MockRProperties_Exist_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Exist'
type MockRProperties_Exist_Call struct {
	*mock.Call
}

// Exist is a helper method to define mock.On call
//   - key string
func (_e *MockRProperties_Expecter) Exist(key interface{}) *MockRProperties_Exist_Call {
	return &MockRProperties_Exist_Call{Call: _e.mock.On("Exist", key)}
}

func (_c *MockRProperties_Exist_Call) Run(run func(key string)) *MockRProperties_Exist_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockRProperties_Exist_Call) Return(_a0 bool) *MockRProperties_Exist_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRProperties_Exist_Call) RunAndReturn(run func(string) bool) *MockRProperties_Exist_Call {
	_c.Call.Return(run)
	return _c
}

// Get provides a mock function with given fields: key
func (_m *MockRProperties) Get(key string) (string, bool) {
	ret := _m.Called(key)

	if len(ret) == 0 {
		panic("no return value specified for Get")
	}

	var r0 string
	var r1 bool
	if rf, ok := ret.Get(0).(func(string) (string, bool)); ok {
		return rf(key)
	}
	if rf, ok := ret.Get(0).(func(string) string); ok {
		r0 = rf(key)
	} else {
		r0 = ret.Get(0).(string)
	}

	if rf, ok := ret.Get(1).(func(string) bool); ok {
		r1 = rf(key)
	} else {
		r1 = ret.Get(1).(bool)
	}

	return r0, r1
}

// MockRProperties_Get_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'Get'
type MockRProperties_Get_Call struct {
	*mock.Call
}

// Get is a helper method to define mock.On call
//   - key string
func (_e *MockRProperties_Expecter) Get(key interface{}) *MockRProperties_Get_Call {
	return &MockRProperties_Get_Call{Call: _e.mock.On("Get", key)}
}

func (_c *MockRProperties_Get_Call) Run(run func(key string)) *MockRProperties_Get_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run(args[0].(string))
	})
	return _c
}

func (_c *MockRProperties_Get_Call) Return(value string, ok bool) *MockRProperties_Get_Call {
	_c.Call.Return(value, ok)
	return _c
}

func (_c *MockRProperties_Get_Call) RunAndReturn(run func(string) (string, bool)) *MockRProperties_Get_Call {
	_c.Call.Return(run)
	return _c
}

// ToRawMap provides a mock function with given fields:
func (_m *MockRProperties) ToRawMap() map[string]string {
	ret := _m.Called()

	if len(ret) == 0 {
		panic("no return value specified for ToRawMap")
	}

	var r0 map[string]string
	if rf, ok := ret.Get(0).(func() map[string]string); ok {
		r0 = rf()
	} else {
		if ret.Get(0) != nil {
			r0 = ret.Get(0).(map[string]string)
		}
	}

	return r0
}

// MockRProperties_ToRawMap_Call is a *mock.Call that shadows Run/Return methods with type explicit version for method 'ToRawMap'
type MockRProperties_ToRawMap_Call struct {
	*mock.Call
}

// ToRawMap is a helper method to define mock.On call
func (_e *MockRProperties_Expecter) ToRawMap() *MockRProperties_ToRawMap_Call {
	return &MockRProperties_ToRawMap_Call{Call: _e.mock.On("ToRawMap")}
}

func (_c *MockRProperties_ToRawMap_Call) Run(run func()) *MockRProperties_ToRawMap_Call {
	_c.Call.Run(func(args mock.Arguments) {
		run()
	})
	return _c
}

func (_c *MockRProperties_ToRawMap_Call) Return(_a0 map[string]string) *MockRProperties_ToRawMap_Call {
	_c.Call.Return(_a0)
	return _c
}

func (_c *MockRProperties_ToRawMap_Call) RunAndReturn(run func() map[string]string) *MockRProperties_ToRawMap_Call {
	_c.Call.Return(run)
	return _c
}

// NewMockRProperties creates a new instance of MockRProperties. It also registers a testing interface on the mock and a cleanup function to assert the mocks expectations.
// The first argument is typically a *testing.T value.
func NewMockRProperties(t interface {
	mock.TestingT
	Cleanup(func())
}) *MockRProperties {
	mock := &MockRProperties{}
	mock.Mock.Test(t)

	t.Cleanup(func() { mock.AssertExpectations(t) })

	return mock
}