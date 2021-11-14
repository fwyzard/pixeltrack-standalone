/* Copyright 2014-2022 Chris Jones, W. David Dagenhart, Matti Kortelainen, Andrea Bocci
 *
 * Originally developed as the ReusableObjectHolder as part of CMSSW.
 *
 * This Source Code Form is subject to the terms of the Apache License, v2.0.
 * If a copy of the license was not distributed with this file, You can obtain one at
 * https://www.apache.org/licenses/LICENSE-2.0 .
 */

#ifndef ObjectCache_h
#define ObjectCache_h

/*
Description: Thread safe way to create and reuse a collection of objects of the same type.

Usage:
This class can be used to safely reuse a series of objects created in advance or on demand.
All member functions of this class (apart from the constructor and destructor) are thread
safe, and can be called concurrently from multiple threads. Thus, the object reuse is safe
across different threads.

The objects returned by the ObjectCache are wrapped in a shared_ptr, that uses a custom
deleter to return them to the cache instead of deleting them. Thus, the lifetime of the
"live" objects must not exceed the lifetime of the ObjectCache object from which they were
obtained.

If the objects returned to the cache may not be immediately ready for reuse (e.g. if they
keep track of some asynchronous operations, and are note synchronised before being returned)
the ObjectCache should be constructed with an "IsReady" functor.
The functor will be used to triage each returned object, and allow their reuse only after
the functor returns true.
If no functor is given, the default behaviour assumes that objects can be immediately reused.

For simplicity, this implementation does not support the use of objects with their own
custom deleter. Support can be added should a use case arise.

The primary way of using the class it to call the get() method. New objects can be
constructed using the default constructor:
\code
  auto objectToUse = cache.get();
  objectToUse->setValue(3);
\endcode
or with a constructor that takes some arguments:
\code
  auto arg = ...;
  auto objectToUse = cache.get(std::in_place, arg);
  objectToUse->setValue(3);
\endcode
or with an arbitrary factory function:
\code
  auto arg = ...;
  auto objectToUse = cache.get([arg]() { return Factory::make(arg); });
  objectToUse->setValue(3);
\endcode
To pass the ownership of the newly created object to the cache, the factory function should return
the object as a raw pointer T* or as a std::unique_ptr<T>.

NOTE: If the client code holds onto the std::shared_ptr<> until another call to the ObjectCache,
it should release the shared_ptr before the call. This way the object can be returned to the
cache and immediately reused. For example:
\code
  std::shared_ptr<MyObject> obj;
  while(someCondition()) {
    //release object so it can re-enter the cache
    obj.release();
    obj = cache.get([]{ return new MyObject();} );
    obj->setValue(someNewValue());
    useTheObject(obj);
  }
\endcode

The example above is very contrived, since a better way would be:
\code
  while(someCondition()) {
    auto obj = cache.get([]{ return new MyObject();} );
    obj->setValue(someNewValue());
    useTheObject(obj);
    //obj goes out of scope and returns the object to the cache
  }
\endcode
*/

#include <atomic>
#include <cassert>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/lockfree/queue.hpp>

namespace internal {

  namespace impl {

    template <typename T>
    struct IsAlwaysTrue {
      bool operator()(T const&) { return true; }
    };

  }  // namespace impl

  template <typename T, typename R = impl::IsAlwaysTrue<T>>
  class ObjectCache {
  public:
    ObjectCache(R canBeReused = R{}, size_t initialCacheSize = 32)
        : m_readyObjects(initialCacheSize),
          m_recycledObjects(0),
          m_outstandingObjects(0),
          m_canBeReused{std::move(canBeReused)} {
      static_assert(std::is_invocable_r<bool, R, T const&>::value);
    }

    ObjectCache(ObjectCache const&) = delete;
    ObjectCache(ObjectCache&&) = delete;
    ObjectCache& operator=(ObjectCache const&) = delete;
    ObjectCache& operator=(ObjectCache&&) = delete;

    ~ObjectCache() {
      assert(0 == m_outstandingObjects);
      m_readyObjects.consume_all([](T* item) { delete item; });
      m_recycledObjects.consume_all([](T* item) { delete item; });
    }

    // Add a non-null item to the cache.
    // Can be used to populate the cache with pre-built objects.
    void add(std::unique_ptr<T> item) {
      if (item != nullptr) {
        if (m_canBeReused(*item)) {
          m_readyObjects.push(getRawPointer(item));
        } else {
          m_recycledObjects.push(getRawPointer(item));
        }
      }
    }

    // Try to get an object from the cache, or construct a new one with the default constructor
    template <typename F>
    std::shared_ptr<T> get() {
      T* item = nullptr;
      if (getReadyObject(item)) {
        return wrapCustomDeleter(item);
      } else {
        return wrapCustomDeleter(new T{});
      }
    }

    // Try to get an object from the cache, or construct a new with the given parameters
    template <typename... Args>
    std::shared_ptr<T> get(std::in_place_t, Args&&... args) {
      T* item = nullptr;
      if (getReadyObject(item)) {
        return wrapCustomDeleter(item);
      } else {
        return wrapCustomDeleter(new T{std::forward<Args>(args)...});
      }
    }

    // Try to get an object from the cache, or create a new one using func
    template <typename F>
    std::shared_ptr<T> get(F func) {
      T* item = nullptr;
      if (getReadyObject(item)) {
        return wrapCustomDeleter(item);
      } else {
        return wrapCustomDeleter(getRawPointer(func()));
      }
    }

  private:
    static T* getRawPointer(T* ptr) { return ptr; }
    static T* getRawPointer(std::unique_ptr<T> ptr) { return ptr.release(); }

    std::shared_ptr<T> wrapCustomDeleter(T* item) {
      // Update the number of "live" objects
      ++m_outstandingObjects;
      // Use a custom deleter that hands the object back to the cache,
      // instead of actually deleting the object
      return std::shared_ptr<T>{item, [this](T* item) { this->addBack(item); }};
    }

    bool getReadyObject(T*& item) {
      // Check if any cached objects is ready, then return it
      if (m_readyObjects.pop(item))
        return true;

      // Otherwise, triage the recycled objects and mark any ready ones
      std::vector<T*> items;
      T* temp = nullptr;
      while (m_recycledObjects.pop(temp)) {
        if (m_canBeReused(*temp)) {
          m_readyObjects.push(temp);
        } else {
          items.push_back(temp);
        }
      }

      // Put the not-yet-ready objects back into the recycled queue
      if (not items.empty()) {
        for (T* temp: items) {
          assert(m_recycledObjects.push(temp));
        }
        items.clear();
      }

      // Check again if any cached objects is ready, then return it
      if (m_readyObjects.pop(item))
        return true;

      return false;
    }

    void addBack(T* item) {
      // If an object is "ready" add it to the cache, otherwise add it to the
      // pool of recylcled objects
      if (m_canBeReused(*item)) {
        m_readyObjects.push(item);
      } else {
        m_recycledObjects.push(item);
      }
      // Update the number of "live" objects
      --m_outstandingObjects;
    }

    boost::lockfree::queue<T*> m_readyObjects;
    boost::lockfree::queue<T*> m_recycledObjects;
    std::atomic<size_t> m_outstandingObjects;

    R m_canBeReused;
  };

}  // namespace internal

#endif  // ObjectCache_h
